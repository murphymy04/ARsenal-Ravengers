"""SQLite database for people, embeddings, and interactions.

Thread-safe via check_same_thread=False. Embeddings stored as
numpy array blobs. All CRUD operations for the AR glasses prototype.
"""

import sqlite3
from datetime import datetime
from typing import List, Optional

import numpy as np

from config import DB_PATH
from models import FaceEmbedding, Person


class SQLiteDatabase:
    """SQLite storage for people, embeddings, and interactions."""

    def __init__(self, db_path: str = str(DB_PATH)):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS people (
                person_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                notes       TEXT DEFAULT '',
                face_thumbnail BLOB,
                is_labeled  INTEGER DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now')),
                last_seen   TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id    INTEGER NOT NULL,
                vector       BLOB NOT NULL,
                model_name   TEXT DEFAULT 'edgeface_xs_gamma_06',
                created_at   TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (person_id) REFERENCES people(person_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id      INTEGER,
                timestamp      TEXT DEFAULT (datetime('now')),
                transcript     TEXT,
                context        TEXT,
                FOREIGN KEY (person_id) REFERENCES people(person_id) ON DELETE SET NULL
            );
        """)
        self._conn.commit()
        self._migrate_schema()

    def _migrate_schema(self):
        """Add columns introduced after initial deployment (idempotent)."""
        try:
            self._conn.execute(
                "ALTER TABLE people ADD COLUMN is_labeled INTEGER DEFAULT 0"
            )
            self._conn.commit()
        except Exception:
            pass

        self._conn.execute("UPDATE people SET is_labeled = 0 WHERE is_labeled IS NULL")
        self._conn.commit()

    def add_person(
        self,
        name: str,
        notes: str = "",
        thumbnail: Optional[np.ndarray] = None,
        is_labeled: bool = True,
    ) -> int:
        """Add a new person. Returns the person_id."""
        thumb_blob = None
        if thumbnail is not None:
            import cv2

            _, buf = cv2.imencode(".jpg", thumbnail)
            thumb_blob = buf.tobytes()

        cur = self._conn.execute(
            "INSERT INTO people (name, notes, face_thumbnail, is_labeled) VALUES (?, ?, ?, ?)",
            (name, notes, thumb_blob, int(is_labeled)),
        )
        self._conn.commit()
        return cur.lastrowid

    def add_auto_person(
        self, thumbnail: Optional[np.ndarray] = None
    ) -> tuple[int, str]:
        """Create an auto-labeled cluster entry. Returns (person_id, auto_name)."""
        person_id = self.add_person(
            "__pending__", is_labeled=False, thumbnail=thumbnail
        )
        auto_name = f"Person {person_id}"
        self.update_person(person_id, name=auto_name)
        return person_id, auto_name

    def get_person(self, person_id: int) -> Optional[Person]:
        """Get a single person by ID."""
        row = self._conn.execute(
            "SELECT person_id, name, notes, face_thumbnail, is_labeled, created_at, last_seen "
            "FROM people WHERE person_id = ?",
            (person_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_person(row)

    def get_person_by_name(self, name: str) -> Optional[Person]:
        """Get the first person with an exact name match, or None."""
        row = self._conn.execute(
            "SELECT person_id, name, notes, face_thumbnail, is_labeled, created_at, last_seen "
            "FROM people WHERE name = ? LIMIT 1",
            (name,),
        ).fetchone()
        return self._row_to_person(row) if row else None

    def get_all_people(self) -> List[Person]:
        """Get all people with their embeddings."""
        rows = self._conn.execute(
            "SELECT person_id, name, notes, face_thumbnail, is_labeled, created_at, last_seen "
            "FROM people ORDER BY name"
        ).fetchall()
        return [self._row_to_person(r) for r in rows]

    def update_person(
        self,
        person_id: int,
        name: str = None,
        notes: str = None,
        is_labeled: bool = None,
    ):
        """Update a person's name, notes, and/or labeled status."""
        if name is not None:
            self._conn.execute(
                "UPDATE people SET name = ? WHERE person_id = ?",
                (name, person_id),
            )
        if notes is not None:
            self._conn.execute(
                "UPDATE people SET notes = ? WHERE person_id = ?",
                (notes, person_id),
            )
        if is_labeled is not None:
            self._conn.execute(
                "UPDATE people SET is_labeled = ? WHERE person_id = ?",
                (int(is_labeled), person_id),
            )
        self._conn.commit()

    def update_last_seen(self, person_id: int):
        """Update last_seen to now."""
        self._conn.execute(
            "UPDATE people SET last_seen = datetime('now') WHERE person_id = ?",
            (person_id,),
        )
        self._conn.commit()

    def delete_person(self, person_id: int):
        """Delete a person and their embeddings (cascade)."""
        self._conn.execute("DELETE FROM people WHERE person_id = ?", (person_id,))
        self._conn.commit()

    def add_embedding(self, person_id: int, embedding: FaceEmbedding):
        """Store a face embedding for a person."""
        blob = embedding.vector.astype(np.float32).tobytes()
        self._conn.execute(
            "INSERT INTO embeddings (person_id, vector, model_name) VALUES (?, ?, ?)",
            (person_id, blob, embedding.model_name),
        )
        self._conn.commit()

    def get_embeddings(self, person_id: int) -> List[FaceEmbedding]:
        """Get all embeddings for a person."""
        rows = self._conn.execute(
            "SELECT vector, model_name FROM embeddings WHERE person_id = ?",
            (person_id,),
        ).fetchall()
        return [
            FaceEmbedding(
                vector=np.frombuffer(r[0], dtype=np.float32).copy(),
                model_name=r[1],
            )
            for r in rows
        ]

    def add_interaction(
        self,
        person_id: Optional[int],
        transcript: str,
        context: str = "",
    ) -> int:
        """Log an interaction. Returns the interaction_id."""
        cur = self._conn.execute(
            "INSERT INTO interactions (person_id, transcript, context) VALUES (?, ?, ?)",
            (person_id, transcript, context),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_interactions(self, person_id: int, limit: int = 20) -> list:
        """Get recent interactions for a person."""
        rows = self._conn.execute(
            "SELECT interaction_id, timestamp, transcript, context "
            "FROM interactions WHERE person_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (person_id, limit),
        ).fetchall()
        return [
            {"id": r[0], "timestamp": r[1], "transcript": r[2], "context": r[3]}
            for r in rows
        ]

    def update_interaction_transcript(
        self, interaction_id: int, transcript: str
    ) -> None:
        """Update the transcript for an interaction."""
        self._conn.execute(
            "UPDATE interactions SET transcript = ? WHERE interaction_id = ?",
            (transcript, interaction_id),
        )
        self._conn.commit()

    def _row_to_person(self, row) -> Person:
        """Convert a database row to a Person object with embeddings."""
        person_id, name, notes, thumb_blob, is_labeled, created_at, last_seen = row

        thumbnail = None
        if thumb_blob:
            import cv2

            arr = np.frombuffer(thumb_blob, dtype=np.uint8)
            thumbnail = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        embeddings = self.get_embeddings(person_id)

        return Person(
            person_id=person_id,
            name=name,
            embeddings=embeddings,
            thumbnail=thumbnail,
            notes=notes or "",
            is_labeled=bool(is_labeled) if is_labeled is not None else True,
            created_at=datetime.fromisoformat(created_at) if created_at else None,
            last_seen=datetime.fromisoformat(last_seen) if last_seen else None,
        )

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
