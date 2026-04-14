"""Configuration for the storage subsystem: Supabase and backend selection."""

import os


STORAGE_BACKEND = os.getenv(
    "STORAGE_BACKEND",
    "sqlite",
)


# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("SUPABASE_PUBLIC_URL")
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TIMEOUT_SECONDS = float(os.getenv("SUPABASE_TIMEOUT_SECONDS", "30"))
