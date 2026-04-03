"""Manual flush from local SQLite to Supabase.

Usage:
    venv/bin/python scripts/migrate_sqlite_to_supabase.py --sqlite-path data/people.db --truncate
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from storage.sqlite_database import SQLiteDatabase
from storage.supabase_database import SupabaseDatabase


def _parse_sqlite_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _load_sqlite_snapshot(sqlite_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")

    people = conn.execute(
        "SELECT person_id, name, notes, face_thumbnail, is_labeled, created_at, last_seen "
        "FROM people ORDER BY person_id"
    ).fetchall()
    embeddings = conn.execute(
        "SELECT embedding_id, person_id, vector, model_name, created_at "
        "FROM embeddings ORDER BY embedding_id"
    ).fetchall()
    interactions = conn.execute(
        "SELECT interaction_id, person_id, timestamp, transcript, context "
        "FROM interactions ORDER BY interaction_id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in people], [dict(r) for r in embeddings], [dict(r) for r in interactions]


def main():
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Flush local SQLite people.db to Supabase")
    parser.add_argument("--sqlite-path", type=Path, default=ROOT / "data" / "people.db")
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Delete existing Supabase rows before importing",
    )
    args = parser.parse_args()

    people_rows, embedding_rows, interaction_rows = _load_sqlite_snapshot(args.sqlite_path)
    remote = SupabaseDatabase()

    if not remote.schema_ready():
        raise SystemExit(
            "Supabase schema is not ready. Apply supabase/schema.sql in the Supabase SQL editor first."
        )

    if args.truncate:
        print("Clearing existing Supabase data...")
        remote.clear_all_data()

    print(f"Importing {len(people_rows)} people...")
    with SQLiteDatabase(str(args.sqlite_path)) as local:
        for row in people_rows:
            person = local.get_person(row["person_id"])
            remote.import_person(
                person_id=person.person_id,
                name=person.name,
                notes=person.notes,
                thumbnail=person.thumbnail,
                is_labeled=person.is_labeled,
                created_at=person.created_at,
                last_seen=person.last_seen,
            )

    print(f"Importing {len(embedding_rows)} embeddings...")
    for row in embedding_rows:
        remote.import_embedding(
            embedding_id=row["embedding_id"],
            person_id=row["person_id"],
            embedding=type("Embedding", (), {
                "vector": np.frombuffer(row["vector"], dtype=np.float32).copy(),
                "model_name": row["model_name"],
            })(),
            created_at=_parse_sqlite_datetime(row["created_at"]),
        )

    print(f"Importing {len(interaction_rows)} interactions...")
    for row in interaction_rows:
        remote.import_interaction(
            interaction_id=row["interaction_id"],
            person_id=row["person_id"],
            timestamp=_parse_sqlite_datetime(row["timestamp"]),
            transcript=row["transcript"] or "",
            context=row["context"] or "",
        )

    people = remote.get_all_people()
    total_embeddings = sum(len(p.embeddings) for p in people)
    print(f"Done. Supabase now has {len(people)} people and {total_embeddings} embeddings.")


if __name__ == "__main__":
    main()
