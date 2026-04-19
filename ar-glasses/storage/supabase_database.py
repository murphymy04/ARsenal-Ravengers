"""Supabase-backed storage implementation.

Uses the Supabase PostgREST API directly so the rest of the application can
keep using the existing synchronous ``Database`` CRUD surface.
"""

import base64
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import requests

from config import (
    EMBEDDING_MODEL_NAME,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_TIMEOUT_SECONDS,
    SUPABASE_URL,
)
from models import FaceEmbedding, Person


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class SupabaseDatabase:
    """Supabase storage for people, embeddings, and interactions."""

    def __init__(
        self,
        _db_path: str | None = None,
        *,
        url: str | None = None,
        key: str | None = None,
        timeout: float = SUPABASE_TIMEOUT_SECONDS,
    ):
        self._url = (url or SUPABASE_URL or "").rstrip("/")
        self._key = key or SUPABASE_SERVICE_ROLE_KEY
        self._timeout = timeout
        if not self._url or not self._key:
            raise ValueError(
                "Supabase backend selected but SUPABASE_PUBLIC_URL and "
                "SUPABASE_SERVICE_ROLE_KEY are not fully configured."
            )
        self._session = requests.Session()
        self._headers = {
            "apikey": self._key,
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        table_path: str,
        *,
        params: dict | None = None,
        json_body=None,
        headers: dict | None = None,
    ):
        response = self._session.request(
            method=method,
            url=f"{self._url}/rest/v1/{table_path}",
            headers={**self._headers, **(headers or {})},
            params=params,
            json=json_body,
            timeout=self._timeout,
        )
        if response.status_code >= 400:
            detail = response.text.strip()
            raise RuntimeError(
                f"Supabase request failed [{response.status_code}]: {detail}"
            )
        if not response.text:
            return None
        return response.json()

    def schema_ready(self) -> bool:
        try:
            self._request("GET", "people", params={"select": "person_id", "limit": 1})
            return True
        except RuntimeError:
            return False

    def clear_all_data(self):
        self._request("DELETE", "interactions", params={"interaction_id": "gt.0"})
        self._request("DELETE", "embeddings", params={"embedding_id": "gt.0"})
        self._request("DELETE", "people", params={"person_id": "gt.0"})

    def _thumbnail_to_base64(self, thumbnail: np.ndarray | None) -> str | None:
        if thumbnail is None:
            return None
        ok, buf = cv2.imencode(".jpg", thumbnail)
        if not ok:
            raise ValueError("Failed to encode thumbnail as JPEG.")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _thumbnail_from_base64(self, payload: str | None) -> np.ndarray | None:
        if not payload:
            return None
        arr = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _embedding_to_list(self, embedding: FaceEmbedding) -> list[float]:
        return embedding.vector.astype(np.float32).tolist()

    def _embedding_from_row(self, row: dict) -> FaceEmbedding:
        return FaceEmbedding(
            vector=np.array(row["vector"], dtype=np.float32),
            model_name=row["model_name"],
        )

    def _person_from_row(self, row: dict, embeddings: list[FaceEmbedding]) -> Person:
        return Person(
            person_id=row["person_id"],
            name=row["name"],
            embeddings=embeddings,
            thumbnail=self._thumbnail_from_base64(row.get("face_thumbnail_base64")),
            notes=row.get("notes") or "",
            is_labeled=bool(row.get("is_labeled", True)),
            created_at=_parse_timestamp(row.get("created_at")),
            last_seen=_parse_timestamp(row.get("last_seen")),
        )

    def add_person(
        self,
        name: str,
        notes: str = "",
        thumbnail: Optional[np.ndarray] = None,
        is_labeled: bool = True,
    ) -> int:
        payload = {
            "name": name,
            "notes": notes,
            "face_thumbnail_base64": self._thumbnail_to_base64(thumbnail),
            "is_labeled": is_labeled,
        }
        rows = self._request(
            "POST",
            "people",
            json_body=payload,
            headers={"Prefer": "return=representation"},
        )
        return rows[0]["person_id"]

    def import_person(
        self,
        *,
        person_id: int,
        name: str,
        notes: str,
        thumbnail: np.ndarray | None,
        is_labeled: bool,
        created_at: datetime | None,
        last_seen: datetime | None,
    ):
        payload = {
            "person_id": person_id,
            "name": name,
            "notes": notes,
            "face_thumbnail_base64": self._thumbnail_to_base64(thumbnail),
            "is_labeled": is_labeled,
            "created_at": created_at.isoformat() if created_at else None,
            "last_seen": last_seen.isoformat() if last_seen else None,
        }
        self._request(
            "POST",
            "people",
            json_body=payload,
            headers={"Prefer": "return=representation,resolution=merge-duplicates"},
        )

    def add_auto_person(
        self, thumbnail: Optional[np.ndarray] = None
    ) -> tuple[int, str]:
        person_id = self.add_person(
            "__pending__", is_labeled=False, thumbnail=thumbnail
        )
        auto_name = f"Person {person_id}"
        self.update_person(person_id, name=auto_name)
        return person_id, auto_name

    def get_person(self, person_id: int) -> Optional[Person]:
        rows = self._request(
            "GET",
            "people",
            params={
                "person_id": f"eq.{person_id}",
                "select": "person_id,name,notes,face_thumbnail_base64,is_labeled,created_at,last_seen",
                "limit": 1,
            },
        )
        if not rows:
            return None
        return self._person_from_row(rows[0], self.get_embeddings(person_id))

    def get_person_by_name(self, name: str) -> Optional[Person]:
        rows = self._request(
            "GET",
            "people",
            params={
                "name": f"eq.{name}",
                "select": "person_id,name,notes,face_thumbnail_base64,is_labeled,created_at,last_seen",
                "limit": 1,
            },
        )
        if not rows:
            return None
        person_id = rows[0]["person_id"]
        return self._person_from_row(rows[0], self.get_embeddings(person_id))

    def get_all_people(self) -> list[Person]:
        people_rows = self._request(
            "GET",
            "people",
            params={
                "select": "person_id,name,notes,face_thumbnail_base64,is_labeled,created_at,last_seen",
                "order": "name.asc",
            },
        )
        embedding_rows = self._request(
            "GET",
            "embeddings",
            params={
                "select": "person_id,vector,model_name",
                "order": "embedding_id.asc",
            },
        )
        by_person: dict[int, list[FaceEmbedding]] = {}
        for row in embedding_rows:
            by_person.setdefault(row["person_id"], []).append(
                self._embedding_from_row(row)
            )
        return [
            self._person_from_row(row, by_person.get(row["person_id"], []))
            for row in people_rows
        ]

    def update_person(
        self,
        person_id: int,
        name: str = None,
        notes: str = None,
        is_labeled: bool = None,
    ):
        payload = {}
        if name is not None:
            payload["name"] = name
        if notes is not None:
            payload["notes"] = notes
        if is_labeled is not None:
            payload["is_labeled"] = is_labeled
        if not payload:
            return
        self._request(
            "PATCH",
            "people",
            params={"person_id": f"eq.{person_id}"},
            json_body=payload,
        )

    def update_last_seen(self, person_id: int):
        self._request(
            "PATCH",
            "people",
            params={"person_id": f"eq.{person_id}"},
            json_body={"last_seen": datetime.utcnow().isoformat()},
        )

    def delete_person(self, person_id: int):
        self._request("DELETE", "people", params={"person_id": f"eq.{person_id}"})

    def add_embedding(self, person_id: int, embedding: FaceEmbedding):
        self._request(
            "POST",
            "embeddings",
            json_body={
                "person_id": person_id,
                "vector": self._embedding_to_list(embedding),
                "model_name": embedding.model_name,
            },
        )

    def import_embedding(
        self,
        *,
        embedding_id: int,
        person_id: int,
        embedding: FaceEmbedding,
        created_at: datetime | None,
    ):
        self._request(
            "POST",
            "embeddings",
            json_body={
                "embedding_id": embedding_id,
                "person_id": person_id,
                "vector": self._embedding_to_list(embedding),
                "model_name": embedding.model_name or EMBEDDING_MODEL_NAME,
                "created_at": created_at.isoformat() if created_at else None,
            },
            headers={"Prefer": "return=representation,resolution=merge-duplicates"},
        )

    def get_embeddings(self, person_id: int) -> list[FaceEmbedding]:
        rows = self._request(
            "GET",
            "embeddings",
            params={
                "person_id": f"eq.{person_id}",
                "select": "vector,model_name",
                "order": "embedding_id.asc",
            },
        )
        return [self._embedding_from_row(row) for row in rows]

    def add_interaction(
        self,
        person_id: Optional[int],
        transcript: str,
        context: str = "",
    ) -> int:
        rows = self._request(
            "POST",
            "interactions",
            json_body={
                "person_id": person_id,
                "transcript": transcript,
                "context": context,
            },
            headers={"Prefer": "return=representation"},
        )
        return rows[0]["interaction_id"]

    def import_interaction(
        self,
        *,
        interaction_id: int,
        person_id: int | None,
        timestamp: datetime | None,
        transcript: str,
        context: str,
    ):
        self._request(
            "POST",
            "interactions",
            json_body={
                "interaction_id": interaction_id,
                "person_id": person_id,
                "timestamp": timestamp.isoformat() if timestamp else None,
                "transcript": transcript,
                "context": context,
            },
            headers={"Prefer": "return=representation,resolution=merge-duplicates"},
        )

    def get_interactions(self, person_id: int, limit: int = 20) -> list:
        rows = self._request(
            "GET",
            "interactions",
            params={
                "person_id": f"eq.{person_id}",
                "select": "interaction_id,timestamp,transcript,context",
                "order": "timestamp.desc",
                "limit": limit,
            },
        )
        return [
            {
                "id": row["interaction_id"],
                "timestamp": row["timestamp"],
                "transcript": row["transcript"],
                "context": row["context"],
            }
            for row in rows
        ]

    def update_interaction_transcript(
        self, interaction_id: int, transcript: str
    ) -> None:
        """Update the transcript for an interaction."""
        self._request(
            "PATCH",
            "interactions",
            json={"transcript": transcript},
            params={"interaction_id": f"eq.{interaction_id}"},
        )

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
