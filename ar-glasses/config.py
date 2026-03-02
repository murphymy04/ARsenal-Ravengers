"""Configuration constants for the AR glasses prototype."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "people.db"
EDGEFACE_ROOT = PROJECT_ROOT / "edgeface"
EDGEFACE_CHECKPOINT = EDGEFACE_ROOT / "checkpoints" / "edgeface_base.pt"

# Face detection (MediaPipe BlazeFace)
DETECTION_CONFIDENCE = 0.5
FACE_CROP_SIZE = 112  # EdgeFace expects 112x112

# Face embedding
EMBEDDING_DIM = 512
EMBEDDING_MODEL_NAME = "edgeface_base"

# Face matching
MATCH_THRESHOLD = 0.4   # cosine similarity threshold for same person
UNKNOWN_LABEL = "Unknown"

# Unsupervised clustering
EMBEDDING_UPDATE_INTERVAL = 60   # frames between accumulating new embeddings for a recognized person
MAX_EMBEDDINGS_PER_PERSON = 20   # cap to prevent unbounded growth per person
MIN_SIGHTINGS_TO_CLUSTER = 8     # consecutive observations needed before promoting to a real cluster
PENDING_CLUSTER_SIMILARITY = 0.25  # cosine similarity threshold for grouping pending observations
PENDING_EXPIRY_FRAMES = 90       # frames of absence before a pending observation is discarded
MERGE_SIMILARITY_THRESHOLD = 0.35  # similarity above which two clusters are suggested for merging

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
