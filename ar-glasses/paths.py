"""Shared filesystem paths used across subsystems.

Kept separate from config.py to avoid circular imports between the top-level
config and per-subsystem config modules.
"""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "people.db"
EDGEFACE_ROOT = PROJECT_ROOT / "edgeface"
EDGEFACE_CHECKPOINT = EDGEFACE_ROOT / "checkpoints" / "edgeface_base.pt"

DATA_DIR.mkdir(exist_ok=True)
