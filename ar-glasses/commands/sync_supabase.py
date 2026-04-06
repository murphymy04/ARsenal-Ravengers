"""Manual sync command for flushing the local SQLite database to Supabase."""

import argparse
import subprocess
import sys
from pathlib import Path


def sync_supabase_mode(sqlite_path: str, truncate: bool = False):
    root = Path(__file__).resolve().parent.parent
    script = root / "scripts" / "migrate_sqlite_to_supabase.py"
    cmd = [sys.executable, str(script), "--sqlite-path", sqlite_path]
    if truncate:
        cmd.append("--truncate")
    subprocess.run(cmd, check=True)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Sync local SQLite people.db to Supabase")
    parser.add_argument("--sqlite-path", type=str, default=str(root / "data" / "people.db"))
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()
    sync_supabase_mode(args.sqlite_path, truncate=args.truncate)


if __name__ == "__main__":
    main()
