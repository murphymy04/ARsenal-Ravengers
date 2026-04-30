"""Database management CLI: list, delete, label, merge, enroll.

Live recognition runs through dashboard.py / pipeline/live.py — this CLI is
just for inspecting and curating the SQLite face database.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Box-drawing chars in command output break Windows cp1252 stdout.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from storage.database import Database


def cmd_db_info(_args):
    from commands.db import db_info_mode

    db = Database()
    try:
        db_info_mode(db)
    finally:
        db.close()


def cmd_db_delete(_args):
    from commands.db import db_delete_mode

    db = Database()
    try:
        db_delete_mode(db)
    finally:
        db.close()


def cmd_label(_args):
    from commands.label import label_mode

    db = Database()
    try:
        label_mode(db)
    finally:
        db.close()


def cmd_merge(_args):
    from commands.merge import merge_clusters_mode

    db = Database()
    try:
        merge_clusters_mode(db)
    finally:
        db.close()


def cmd_enroll(_args):
    from commands.enroll import enroll_mode
    from processing.face_detector import FaceDetector
    from processing.face_embedder import FaceEmbedder

    db = Database()
    detector = FaceDetector()
    embedder = FaceEmbedder()
    try:
        enroll_mode(db, detector, embedder)
    finally:
        detector.close()
        db.close()


def main():
    parser = argparse.ArgumentParser(
        prog="cli", description="ARsenal face database tools"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("db-info", help="List everyone in the database").set_defaults(
        func=cmd_db_info
    )
    sub.add_parser(
        "db-delete", help="Remove a person or wipe the database"
    ).set_defaults(func=cmd_db_delete)
    sub.add_parser("label", help="Assign names to auto-discovered clusters").set_defaults(
        func=cmd_label
    )
    sub.add_parser("merge", help="Find and merge duplicate clusters").set_defaults(
        func=cmd_merge
    )
    sub.add_parser(
        "enroll", help="Manually enroll a named person via webcam"
    ).set_defaults(func=cmd_enroll)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
