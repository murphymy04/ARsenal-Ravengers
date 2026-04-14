"""Top-level configuration.

Loads `.env` (via `paths.py`), defines cross-cutting constants (Flask host/port,
display colors) and re-exports every per-subsystem constant so existing
`from config import X` call sites keep working.
"""

from paths import (  # noqa: F401
    DATA_DIR,
    DB_PATH,
    EDGEFACE_CHECKPOINT,
    EDGEFACE_ROOT,
    PROJECT_ROOT,
)

from input.config import *  # noqa: F401, F403
from pipeline.config import *  # noqa: F401, F403
from processing.config import *  # noqa: F401, F403
from storage.config import *  # noqa: F401, F403


# Display
BBOX_COLOR = (0, 255, 0)  # green
UNKNOWN_BBOX_COLOR = (0, 0, 255)  # red
TEXT_COLOR = (255, 255, 255)  # white
FONT_SCALE = 0.6
FONT_THICKNESS = 2


# Companion app
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
