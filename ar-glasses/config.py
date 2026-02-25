"""Configuration constants for the AR glasses prototype."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "people.db"
EDGEFACE_ROOT = PROJECT_ROOT / "edgeface"
EDGEFACE_CHECKPOINT = EDGEFACE_ROOT / "checkpoints" / "edgeface_xs_gamma_06.pt"

# Face detection (OpenCV Haar Cascade)
FACE_CROP_SIZE = 112  # EdgeFace expects 112x112

# Face embedding
EMBEDDING_DIM = 512
EMBEDDING_MODEL_NAME = "edgeface_xs_gamma_06"

# Face matching
MATCH_THRESHOLD = 0.4   # cosine similarity threshold for same person
UNKNOWN_LABEL = "Unknown"

# Camera
CAMERA_SOURCE = 0       # default webcam
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Microphone
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5    # seconds per audio chunk
VAD_AGGRESSIVENESS = 2  # webrtcvad aggressiveness (0-3)

# Audio processing
WHISPER_MODEL = "small"
WHISPER_LANGUAGE = "en"
SILENCE_THRESHOLD = 2.0  # seconds of silence before processing

# Display
BBOX_COLOR = (0, 255, 0)       # green
UNKNOWN_BBOX_COLOR = (0, 0, 255)  # red
TEXT_COLOR = (255, 255, 255)    # white
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Companion app
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)
