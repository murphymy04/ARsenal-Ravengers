"""Storage backend selector.

Imports ``Database`` from the configured backend so the rest of the codebase
can keep using ``from storage.database import Database``.
"""

from config import STORAGE_BACKEND

if STORAGE_BACKEND == "supabase":
    from storage.supabase_database import SupabaseDatabase as Database
else:
    from storage.sqlite_database import SQLiteDatabase as Database

