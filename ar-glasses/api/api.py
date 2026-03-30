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

import io
import json
import base64
from typing import Optional, List
from datetime import datetime
import os

try:
    from fastapi import FastAPI, HTTPException, Body, Depends, Header
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("FastAPI required for API. Install: pip install fastapi uvicorn python-multipart")

import cv2
import numpy as np
from storage.database import Database
from models import Person
from config import FIREBASE_CREDENTIALS, FIREBASE_PROJECT_ID

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth, credentials
    from firebase_admin.exceptions import FirebaseError
except ImportError:
    firebase_admin = None


def _init_firebase():
    """Initialize Firebase Admin SDK once."""
    if not firebase_admin:
        raise RuntimeError("firebase_admin package is required for Firebase token validation")
    if firebase_admin._apps:
        return

    if not FIREBASE_CREDENTIALS:
        raise RuntimeError("FIREBASE_CREDENTIALS env var is required for Firebase auth")

    # support either JSON path or JSON object string
    try:
        if FIREBASE_CREDENTIALS.strip().startswith("{"):
            data = FIREBASE_CREDENTIALS
            cred = credentials.Certificate(json.loads(data))
        else:
            cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Firebase credentials: {exc}. "
                          f"Make sure FIREBASE_CREDENTIALS points to a valid service account key file "
                          f"(not firebase.json config file). Download from Firebase Console > "
                          f"Project Settings > Service Accounts > Generate new private key.")

    # Validate it's actually a service account key
    try:
        if FIREBASE_CREDENTIALS.strip().startswith("{"):
            parsed = json.loads(FIREBASE_CREDENTIALS)
        else:
            with open(FIREBASE_CREDENTIALS, 'r') as f:
                parsed = json.load(f)

        if parsed.get('type') != 'service_account':
            raise RuntimeError("Firebase credentials file is not a service account key. "
                             "It appears to be a client config file. Please download the "
                             "service account key from Firebase Console > Project Settings > "
                             "Service Accounts > Generate new private key.")
    except Exception as exc:
        if "not a service account key" in str(exc):
            raise exc
        raise RuntimeError(f"Failed to validate Firebase credentials format: {exc}")

    kwargs = {}
    if FIREBASE_PROJECT_ID:
        kwargs["projectId"] = FIREBASE_PROJECT_ID

    firebase_admin.initialize_app(cred, kwargs)


def _get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """Require Bearer token in Authorization header and validate against Firebase."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Use 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]

    try:
        _init_firebase()
        decoded = firebase_auth.verify_id_token(token)
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decoded.get("uid")


# =============================================================================
# Request/Response Models
# =============================================================================

class PersonResponse(BaseModel):
    """A person as returned by the API (includes metadata, not full embeddings)."""
    person_id: int
    name: str
    is_labeled: bool
    embedding_count: int
    notes: str
    created_at: str
    last_seen: Optional[str]
    thumbnail_url: Optional[str] = None  # e.g., "/api/people/42/thumbnail"

    class Config:
        from_attributes = True


class UnlabeledResponse(BaseModel):
    """Unlabeled cluster awaiting a name assignment."""
    person_id: int
    name: str  # auto-generated, e.g. "Person 42"
    embedding_count: int
    last_seen: Optional[str]
    thumbnail_url: Optional[str]


class LabelRequest(BaseModel):
    """Assign a name to an unlabeled cluster."""
    name: str


class MergeRequest(BaseModel):
    """Merge two clusters, keeping embeddings from both."""
    keep_person_id: int
    discard_person_id: int


class LabelResponse(BaseModel):
    """Response after labeling."""
    person_id: int
    name: str
    is_labeled: bool
    action: str  # "labeled" or "merged"
    details: Optional[str] = None


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
            dependencies=[Depends(_get_current_user)],
        )
        self._setup_routes()

    def _setup_routes(self):
        """Register all API endpoints."""

        @self.app.get("/", tags=["health"])
        def health():
            """Health check."""
            return {"status": "ok", "service": "ar-glasses-labeling-api"}

        @self.app.get("/api/people", response_model=List[PersonResponse], tags=["people"])
        def get_all_people():
            """Get all enrolled people with metadata (including labeled and unlabeled)."""
            people = self.db.get_all_people()
            return [self._person_to_response(p) for p in people]

        @self.app.get("/api/people/unlabeled", response_model=List[UnlabeledResponse], tags=["labeling"])
        def get_unlabeled_clusters():
            """Get only unlabeled clusters awaiting names from the mobile app."""
            people = self.db.get_all_people()
            unlabeled = [p for p in people if not p.is_labeled]
            return [
                UnlabeledResponse(
                    person_id=p.person_id,
                    name=p.name,
                    embedding_count=len(p.embeddings),
                    last_seen=p.last_seen.isoformat() if p.last_seen else None,
                    thumbnail_url=f"/api/people/{p.person_id}/thumbnail" if p.thumbnail is not None else None,
                )
                for p in unlabeled
            ]

        @self.app.get("/api/people/{person_id}", response_model=PersonResponse, tags=["people"])
        def get_person(person_id: int):
            """Get details for a single person."""
            person = self.db.get_person(person_id)
            if not person:
                raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
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
                raise HTTPException(status_code=404, detail=f"No thumbnail for person {person_id}")

            if format == "base64":
                # Return as base64-encoded data URL for mobile app embed
                _, buf = cv2.imencode(".jpg", person.thumbnail)
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                return JSONResponse({
                    "person_id": person_id,
                    "name": person.name,
                    "thumbnail_data_url": f"data:image/jpeg;base64,{b64}"
                })
            else:
                # Return as JPEG stream
                _, buf = cv2.imencode(".jpg", person.thumbnail)
                return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")

        @self.app.post("/api/people/{person_id}/label", response_model=LabelResponse, tags=["labeling"])
        def label_person(person_id: int, request: LabelRequest = Body(...)):
            """Assign a name to an unlabeled cluster.

            If the name matches an existing person, merges the clusters.
            Otherwise, marks this cluster as labeled with the new name.
            """
            person = self.db.get_person(person_id)
            if not person:
                raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

            if person.is_labeled:
                raise HTTPException(
                    status_code=400,
                    detail=f"Person {person_id} is already labeled; cannot relabel"
                )

            name = request.name.strip()
            if not name:
                raise HTTPException(status_code=400, detail="Name cannot be empty")

            # Check if name already exists
            existing = self.db.get_person_by_name(name)

            if existing and existing.person_id != person_id:
                # Merge case: move embeddings from person → existing
                self._merge_people(keep=existing, discard=person)
                return LabelResponse(
                    person_id=existing.person_id,
                    name=existing.name,
                    is_labeled=True,
                    action="merged",
                    details=f"Cluster {person_id} merged into existing person {existing.person_id}"
                )
            else:
                # New name: update person and mark as labeled
                self.db.update_person(person_id, name=name, is_labeled=True)
                return LabelResponse(
                    person_id=person_id,
                    name=name,
                    is_labeled=True,
                    action="labeled",
                    details=f"Cluster {person_id} labeled as '{name}'"
                )

        @self.app.post("/api/people/merge", response_model=LabelResponse, tags=["admin"])
        def merge_clusters(request: MergeRequest = Body(...)):
            """Merge two clusters (admin tool).

            Keeps all embeddings from both clusters.
            Deletes the discard_person_id entry.
            """
            keep = self.db.get_person(request.keep_person_id)
            discard = self.db.get_person(request.discard_person_id)

            if not keep or not discard:
                raise HTTPException(status_code=404, detail="One or both people not found")

            if keep.person_id == discard.person_id:
                raise HTTPException(status_code=400, detail="Cannot merge person with themselves")

            self._merge_people(keep=keep, discard=discard)
            return LabelResponse(
                person_id=keep.person_id,
                name=keep.name,
                is_labeled=keep.is_labeled,
                action="merged",
                details=f"Merged person {discard.person_id} into {keep.person_id}"
            )

        @self.app.delete("/api/people/{person_id}", tags=["admin"])
        def delete_person(person_id: int):
            """Delete a person and all their embeddings (irreversible)."""
            person = self.db.get_person(person_id)
            if not person:
                raise HTTPException(status_code=404, detail=f"Person {person_id} not found")

            self.db.delete_person(person_id)
            return JSONResponse({
                "person_id": person_id,
                "name": person.name,
                "status": "deleted"
            })

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
            thumbnail_url=f"/api/people/{person.person_id}/thumbnail" if person.thumbnail is not None else None,
        )

    def _merge_people(self, keep: Person, discard: Person):
        """Move all embeddings from discard → keep, then delete discard."""
        for emb in discard.embeddings:
            self.db.add_embedding(keep.person_id, emb)
        self.db.update_last_seen(keep.person_id)
        self.db.delete_person(discard.person_id)

    def run(self, host: str = "0.0.0.0", port: int = 5000, reload: bool = False):
        """Start the API server.

        Args:
            host: bind address
            port: port number
            reload: auto-reload on code changes (dev only)
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, reload=reload)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from config import DB_PATH, FLASK_HOST, FLASK_PORT

    parser = argparse.ArgumentParser(description="AR Glasses Labeling API")
    parser.add_argument("--db", type=str, default=str(DB_PATH), help="Path to people.db")
    parser.add_argument("--host", type=str, default=FLASK_HOST, help="Bind address")
    parser.add_argument("--port", type=int, default=FLASK_PORT, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes (dev)")
    args = parser.parse_args()

    db = Database(args.db)
    api = PeopleAPI(db)

    print(f"Starting API on {args.host}:{args.port}")
    print(f"Database: {args.db}")
    print(f"OpenAPI docs: http://{args.host}:{args.port}/docs")
    api.run(host=args.host, port=args.port, reload=args.reload)
