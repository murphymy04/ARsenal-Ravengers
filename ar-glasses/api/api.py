"""REST API for companion mobile app — labeling & people management.

Provides endpoints for:
- Getting all people (with embeddings count, last_seen, thumbnails)
- Getting only unlabeled clusters (awaiting assignment)
- Labeling a cluster with a name
- Merging two clusters (admin tool)
- Serving thumbnail images

Runs independently of the video pipeline; can be started in parallel.
Uses FastAPI + Uvicorn for async performance.

Usage:
    api = PeopleAPI(db)
    api.run(host="0.0.0.0", port=5000)
    # or from CLI: python -m ar-glasses.api --db data/people.db --port 5000
"""

import base64
import io

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as e:
    raise ImportError(
        "FastAPI required for API. Install: pip install fastapi uvicorn "
        "python-multipart"
    ) from e

import cv2
from models import Person
from storage.database import Database

from api.requests import (
    InteractionRequest,
    LabelRequest,
    MergeRequest,
    NotesRequest,
)
from api.responses import (
    InteractionResponse,
    LabelResponse,
    PersonResponse,
    UnlabeledResponse,
)

try:
    from pipeline.knowledge import flush_memory, save_to_memory
except ImportError:
    save_to_memory = None
    flush_memory = None

# Request/Response Models are in api/responses/ and api/requests/


# =============================================================================
# API Implementation
# =============================================================================


class PeopleAPI:
    """REST API for face labeling & people management."""

    def __init__(self, db: Database):
        self.db = db
        self.app = FastAPI(
            title="AR Glasses Labeling API",
            description="Companion app backend for labeling face clusters",
            version="1.0",
        )
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register all API endpoints."""

        @self.app.get("/", tags=["health"])
        def health():
            """Health check."""
            return {"status": "ok", "service": "ar-glasses-labeling-api"}

        @self.app.get(
            "/api/people", response_model=list[PersonResponse], tags=["people"]
        )
        def get_all_people():
            """Get all enrolled people with metadata."""
            people = self.db.get_all_people()
            # Filter out duplicates: keep only first person per name (case/whitespace insensitive)
            seen_names = set()
            unique_people = []
            for p in people:
                norm_name = self._normalize_name(p.name)
                if norm_name not in seen_names:
                    seen_names.add(norm_name)
                    unique_people.append(p)
            return [self._person_to_response(p) for p in unique_people]

        @self.app.get(
            "/api/people/unlabeled",
            response_model=list[UnlabeledResponse],
            tags=["labeling"],
        )
        def get_unlabeled_clusters():
            """Get only unlabeled clusters awaiting names from the mobile app."""
            people = self.db.get_all_people()
            unlabeled = [p for p in people if not p.is_labeled]
            results = []
            for p in unlabeled:
                thumbnail_b64 = None
                if p.thumbnail is not None:
                    try:
                        _, buffer = cv2.imencode(".jpg", p.thumbnail)
                        thumbnail_b64 = base64.b64encode(buffer.tobytes()).decode(
                            "utf-8"
                        )
                    except Exception as e:
                        print(f"Error encoding thumbnail for person {p.person_id}: {e}")

                results.append(
                    UnlabeledResponse(
                        person_id=p.person_id,
                        name=p.name,
                        embedding_count=len(p.embeddings),
                        last_seen=p.last_seen.isoformat() if p.last_seen else None,
                        thumbnail=thumbnail_b64,
                    )
                )
            return results

        @self.app.get(
            "/api/people/labeled",
            response_model=list[PersonResponse],
            tags=["people"],
        )
        def get_labeled_people():
            """Get all labeled people with metadata."""
            people = self.db.get_all_people()
            labeled = [p for p in people if p.is_labeled]

            # Filter out duplicates: keep only first person per name (case/whitespace insensitive)
            seen_names = set()
            unique_people = []
            for p in labeled:
                norm_name = self._normalize_name(p.name)
                if norm_name not in seen_names:
                    seen_names.add(norm_name)
                    unique_people.append(p)

            results = []
            for p in unique_people:
                thumbnail_b64 = None
                if p.thumbnail is not None:
                    try:
                        _, buffer = cv2.imencode(".jpg", p.thumbnail)
                        thumbnail_b64 = base64.b64encode(buffer.tobytes()).decode(
                            "utf-8"
                        )
                    except Exception as e:
                        print(f"Error encoding thumbnail for person {p.person_id}: {e}")

                results.append(
                    PersonResponse(
                        person_id=p.person_id,
                        name=p.name,
                        is_labeled=p.is_labeled,
                        embedding_count=len(p.embeddings),
                        notes=p.notes,
                        created_at=p.created_at.isoformat() if p.created_at else "",
                        last_seen=p.last_seen.isoformat() if p.last_seen else None,
                        thumbnail=thumbnail_b64,
                    )
                )

            return results

        @self.app.get(
            "/api/people/{person_id}", response_model=PersonResponse, tags=["people"]
        )
        def get_person(person_id: int):
            """Get details for a single person."""
            person = self.db.get_person(person_id)
            if not person:
                raise HTTPException(
                    status_code=404, detail=f"Person {person_id} not found"
                )
            return self._person_to_response(person)

        @self.app.get("/api/people/{person_id}/thumbnail", tags=["people"])
        def get_thumbnail(person_id: int, format: str = "jpeg"):
            """Serve person's thumbnail image as JPEG or base64-encoded data URL.

            Query params:
              - format: "jpeg" (default, streams JPEG bytes)
                        "base64" (returns JSON with base64-encoded image)
            """
            person = self.db.get_person(person_id)
            if not person or person.thumbnail is None:
                raise HTTPException(
                    status_code=404, detail=f"No thumbnail for person {person_id}"
                )

            if format == "base64":
                # Return as base64-encoded data URL for mobile app embed
                _, buf = cv2.imencode(".jpg", person.thumbnail)
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                return JSONResponse(
                    {
                        "person_id": person_id,
                        "name": person.name,
                        "thumbnail_data_url": f"data:image/jpeg;base64,{b64}",
                    }
                )
            else:
                # Return as JPEG stream
                _, buf = cv2.imencode(".jpg", person.thumbnail)
                return StreamingResponse(
                    io.BytesIO(buf.tobytes()), media_type="image/jpeg"
                )

        @self.app.post(
            "/api/people/{person_id}/label",
            response_model=LabelResponse,
            tags=["labeling"],
        )
        def label_person(person_id: int, request: LabelRequest):
            """Assign a name to an unlabeled cluster.

            Marks this cluster as labeled with the new name.
            Does not merge with existing people.
            """
            person = self.db.get_person(person_id)
            if not person:
                raise HTTPException(
                    status_code=404, detail=f"Person {person_id} not found"
                )

            if person.is_labeled:
                raise HTTPException(
                    status_code=400,
                    detail=f"Person {person_id} is already labeled; cannot relabel",
                )

            name = request.name.strip()
            if not name:
                raise HTTPException(status_code=400, detail="Name cannot be empty")

            # Update person with the new name and mark as labeled
            self.db.update_person(person_id, name=name, is_labeled=True)
            # Update interactions to use the newly labeled name
            self._update_interaction_speaker_names(person_id, name)
            # Flush updated interactions to Zep
            self._flush_person_interactions_to_zep(person_id, name)
            return LabelResponse(
                person_id=person_id,
                name=name,
                is_labeled=True,
                action="labeled",
                details=f"Cluster {person_id} labeled as '{name}'",
            )

        @self.app.post(
            "/api/people/{person_id}/notes",
            response_model=PersonResponse,
            tags=["people"],
        )
        def update_notes(person_id: int, request: NotesRequest):
            """Update notes for a specific person."""
            person = self.db.get_person(person_id)
            if not person:
                raise HTTPException(
                    status_code=404, detail=f"Person {person_id} not found"
                )

            self.db.update_person(person_id, notes=request.notes)
            updated_person = self.db.get_person(person_id)
            return self._person_to_response(updated_person)

        @self.app.post(
            "/api/people/merge", response_model=LabelResponse, tags=["admin"]
        )
        def merge_clusters(request: MergeRequest):
            """Merge two clusters (admin tool).

            Keeps all embeddings from both clusters.
            Deletes the discard_person_id entry.
            """
            keep = self.db.get_person(request.keep_person_id)
            discard = self.db.get_person(request.discard_person_id)

            if not keep or not discard:
                raise HTTPException(
                    status_code=404, detail="One or both people not found"
                )

            if keep.person_id == discard.person_id:
                raise HTTPException(
                    status_code=400, detail="Cannot merge person with themselves"
                )

            self._merge_people(keep=keep, discard=discard)
            return LabelResponse(
                person_id=keep.person_id,
                name=keep.name,
                is_labeled=keep.is_labeled,
                action="merged",
                details=f"Merged person {discard.person_id} into {keep.person_id}",
            )

        @self.app.delete("/api/people/{person_id}", tags=["admin"])
        def delete_person(person_id: int):
            """Delete a person and all their embeddings (irreversible)."""
            person = self.db.get_person(person_id)
            if not person:
                raise HTTPException(
                    status_code=404, detail=f"Person {person_id} not found"
                )

            self.db.delete_person(person_id)
            return JSONResponse(
                {"person_id": person_id, "name": person.name, "status": "deleted"}
            )

        # =================================================================
        # Interaction Routes
        # =================================================================

        @self.app.get(
            "/api/interactions",
            response_model=list[InteractionResponse],
            tags=["interactions"],
        )
        def get_all_interactions():
            """Get all interactions from all people.

            Maps interaction person_ids to primary person_id for each name
            so the frontend can associate interactions with the correct person entity.
            """
            # Build mapping of person_id -> primary person_id for each name
            person_id_mapping = self._get_primary_person_id_mapping()

            rows = self.db._conn.execute(
                "SELECT i.interaction_id, i.person_id, i.timestamp, i.transcript, i.context, p.name "
                "FROM interactions i "
                "LEFT JOIN people p ON i.person_id = p.person_id "
                "ORDER BY i.timestamp DESC"
            ).fetchall()
            return [
                InteractionResponse(
                    interaction_id=r[0],
                    person_id=person_id_mapping.get(
                        r[1], r[1]
                    ),  # Map to primary person_id
                    timestamp=r[2],
                    transcript=r[3],
                    context=r[4],
                    person_name=r[5],
                )
                for r in rows
            ]

        @self.app.get(
            "/api/interactions/labeled",
            response_model=list[InteractionResponse],
            tags=["interactions"],
        )
        def get_labeled_interactions():
            """Get all interactions from labeled people only.

            Maps interaction person_ids to primary person_id for each name
            so the frontend can associate interactions with the correct person entity.
            """
            # Build mapping of person_id -> primary person_id for each name
            person_id_mapping = self._get_primary_person_id_mapping()

            rows = self.db._conn.execute(
                "SELECT i.interaction_id, i.person_id, i.timestamp, i.transcript, i.context, p.name "
                "FROM interactions i "
                "LEFT JOIN people p ON i.person_id = p.person_id "
                "WHERE p.is_labeled = 1 "
                "ORDER BY i.timestamp DESC"
            ).fetchall()
            return [
                InteractionResponse(
                    interaction_id=r[0],
                    person_id=person_id_mapping.get(
                        r[1], r[1]
                    ),  # Map to primary person_id
                    timestamp=r[2],
                    transcript=r[3],
                    context=r[4],
                    person_name=r[5],
                )
                for r in rows
            ]

        @self.app.get(
            "/api/people/{person_id}/interactions",
            response_model=list[InteractionResponse],
            tags=["interactions"],
        )
        def get_person_interactions(person_id: int, limit: int = 20):
            """Get recent interactions for a specific person."""
            person = self.db.get_person(person_id)
            if not person:
                raise HTTPException(
                    status_code=404, detail=f"Person {person_id} not found"
                )
            interactions = self.db.get_interactions(person_id, limit=limit)
            return [
                InteractionResponse(
                    interaction_id=i["id"],
                    person_id=person_id,
                    timestamp=i["timestamp"],
                    transcript=i["transcript"],
                    context=i["context"],
                )
                for i in interactions
            ]

        @self.app.post(
            "/api/interactions",
            response_model=InteractionResponse,
            tags=["interactions"],
        )
        def create_interaction(request: InteractionRequest):
            """Log a new interaction."""
            interaction_id = self.db.add_interaction(
                person_id=request.person_id,
                transcript=request.transcript,
                context=request.context,
            )
            interaction = self.db._conn.execute(
                "SELECT interaction_id, person_id, timestamp, transcript, context "
                "FROM interactions WHERE interaction_id = ?",
                (interaction_id,),
            ).fetchone()
            return InteractionResponse(
                interaction_id=interaction[0],
                person_id=interaction[1],
                timestamp=interaction[2],
                transcript=interaction[3],
                context=interaction[4],
            )

    def _person_to_response(self, person: Person) -> PersonResponse:
        """Convert Person dataclass to API response."""
        return PersonResponse(
            person_id=person.person_id,
            name=person.name,
            is_labeled=person.is_labeled,
            embedding_count=len(person.embeddings),
            notes=person.notes,
            created_at=person.created_at.isoformat() if person.created_at else "",
            last_seen=person.last_seen.isoformat() if person.last_seen else None,
            thumbnail_url=f"/api/people/{person.person_id}/thumbnail"
            if person.thumbnail is not None
            else None,
        )

    def _normalize_name(self, name: str) -> str:
        """Normalize name for duplicate comparison: lowercase and collapse whitespace.

        Args:
            name: The name to normalize

        Returns:
            Normalized name (lowercase, single spaces)
        """
        return " ".join(name.lower().split())

    def _get_primary_person_id_mapping(self) -> dict:
        people = self.db.get_all_people()
        mapping = {}
        name_to_primary_id = {}

        # Sort by person_id to ensure lower IDs are primary
        for person in sorted(people, key=lambda p: p.person_id):
            if person.is_labeled:
                # First labeled person becomes primary
                if person.name not in name_to_primary_id:
                    name_to_primary_id[person.name] = person.person_id

            # Only map if a labeled primary exists
            if person.name in name_to_primary_id:
                mapping[person.person_id] = name_to_primary_id[person.name]
            else:
                mapping[person.person_id] = person.person_id
        return mapping

    def _merge_people(self, keep: Person, discard: Person) -> None:
        """Move all embeddings from discard → keep, then delete discard."""
        for emb in discard.embeddings:
            self.db.add_embedding(keep.person_id, emb)
        self.db.update_last_seen(keep.person_id)
        self.db.delete_person(discard.person_id)

    def _update_interaction_speaker_names(self, person_id: int, new_name: str) -> None:
        """Update interaction transcripts to use the newly labeled person's name.

        Replaces non-Wearer speaker names with the new_name in all interactions
        for the given person_id.

        Args:
            person_id: The person whose interactions to update
            new_name: The new name to use for the person in transcripts
        """
        interactions = self.db.get_interactions(person_id, limit=1000)

        for interaction in interactions:
            transcript = interaction["transcript"]
            lines = transcript.split("\n")
            updated_lines = []

            for line in lines:
                # Replace any non-Wearer speaker with the new_name
                if ": " in line:
                    speaker, text = line.split(": ", 1)
                    if speaker != "Wearer":
                        # Replace old speaker name with new name
                        line = f"{new_name}: {text}"
                updated_lines.append(line)

            updated_transcript = "\n".join(updated_lines)

            # Only update if transcript changed
            if updated_transcript != transcript:
                self.db.update_interaction_transcript(
                    interaction["id"], updated_transcript
                )

    def _flush_person_interactions_to_zep(
        self, person_id: int, person_name: str
    ) -> None:
        """Flush all stored interactions for a newly labeled person to Graphiti.

        Mirrors ``LivePipelineDriver._save_conversation``: each interaction's
        transcript is dispatched via ``save_to_memory`` with the resolved
        name, then ``flush_memory`` blocks until pending episodes land in the
        knowledge graph.
        """
        if save_to_memory is None or flush_memory is None:
            print("  [knowledge] skipping Zep flush — knowledge support unavailable")
            return

        interactions = self.db.get_interactions(person_id, limit=1000)
        if not interactions:
            print(f"  [knowledge] no interactions to flush for {person_name}")
            return

        flushed = 0
        for interaction in interactions:
            transcript = interaction["transcript"]
            if not transcript:
                continue
            save_to_memory(transcript, other_name=person_name)
            flushed += 1

        if not flushed:
            print(f"  [knowledge] no non-empty transcripts for {person_name}")
            return

        print(f"[knowledge] waiting for pending saves for {person_name}...")
        flush_memory()
        print(f"[knowledge] flushed {flushed} interactions to Zep for {person_name}")

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        reload: bool = False,
    ) -> None:
        """Start the API server.

        Args:
            host: bind address
            port: port number
            reload: auto-reload on code changes (dev only)
        """
        import uvicorn

        uvicorn.run(self.app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse

    from config import DB_PATH, FLASK_HOST, FLASK_PORT

    parser = argparse.ArgumentParser(description="AR Glasses Labeling API")
    parser.add_argument(
        "--db", type=str, default=str(DB_PATH), help="Path to people.db"
    )
    parser.add_argument("--host", type=str, default=FLASK_HOST, help="Bind address")
    parser.add_argument("--port", type=int, default=FLASK_PORT, help="Port number")
    parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on changes (dev)"
    )
    args = parser.parse_args()

    db = Database(args.db)
    api = PeopleAPI(db)

    print(f"Starting API on {args.host}:{args.port}")
    print(f"Database: {args.db}")
    print(f"OpenAPI docs: http://{args.host}:{args.port}/docs")
    api.run(host=args.host, port=args.port, reload=args.reload)
