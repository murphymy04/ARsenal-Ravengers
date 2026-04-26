"""Storage backend.

Re-exports ``Database`` so the rest of the codebase can keep using
``from storage.database import Database``.
"""

from storage.sqlite_database import SQLiteDatabase as Database  # noqa: F401
