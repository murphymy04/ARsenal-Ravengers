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
# "short_range" — Tasks API, optimised for faces < ~2 m (selfie-style)
# "full_range"  — Solutions API, handles faces up to ~5 m (back-camera style)
FACE_DETECTOR_MODEL = "opencv"
DETECTION_CONFIDENCE = 0.35
FACE_CROP_SIZE = 112  # EdgeFace expects 112x112

# Face embedding
EMBEDDING_DIM = 512
EMBEDDING_MODEL_NAME = "edgeface_base"

# Face matching
MATCH_THRESHOLD = 0.4  # cosine similarity threshold for same person
UNKNOWN_LABEL = "Unknown"

# Unsupervised clustering
EMBEDDING_UPDATE_INTERVAL = (
    60  # frames between accumulating new embeddings for a recognized person
)
MAX_EMBEDDINGS_PER_PERSON = 20  # cap to prevent unbounded growth per person
EMBEDDING_DIVERSITY_THRESHOLD = (
    0.15  # min cosine distance required to store a new embedding (#8)
)
MIN_SIGHTINGS_TO_CLUSTER = 8  # observations needed before promoting to a real cluster
PENDING_CLUSTER_SIMILARITY = 0.25  # cosine similarity for grouping pending observations
PENDING_EXPIRY_FRAMES = 90  # frames of absence before discarding a pending observation
MERGE_SIMILARITY_THRESHOLD = (
    0.35  # similarity above which two clusters are suggested for merging
)

# Face quality (#1, #2)
FACE_MIN_SIZE = 20  # px — ignore detections smaller than this (width or height)
FACE_BLUR_THRESHOLD = 15.0  # Laplacian variance — lower for H.264 compressed video

# Speaking detection backend
# Set SPEAKING_BACKEND = "light_asd" to use Light-ASD (audio-visual, more accurate).
# Set SPEAKING_BACKEND = "mediapipe" to use the MediaPipe jawOpen blendshape (visual-only).
# Set SPEAKING_BACKEND = "vad_rms" to use Silero VAD + RMS amplitude (wearer vs other).
SPEAKING_BACKEND = "vad_rms"

# MediaPipe FaceLandmarker jawOpen blendshape (used when SPEAKING_BACKEND = "mediapipe")
SPEAKING_JAW_THRESHOLD = (
    0.005  # jawOpen score above which the person is considered speaking
)

# Light-ASD (used when SPEAKING_BACKEND = "light_asd")
LIGHT_ASD_WEIGHTS = (
    DATA_DIR / "light_asd.model"
)  # downloaded automatically on first run
LIGHT_ASD_VIDEO_FRAMES = (
    30  # rolling window of face crops per inference (≈1 s at 30 FPS)
)
LIGHT_ASD_MIN_FRAMES = 10  # minimum buffered frames before running inference
LIGHT_ASD_INFERENCE_INTERVAL = 5  # run inference every N video frames
LIGHT_ASD_SPEAKING_THRESHOLD = 0.25  # softmax probability above which = speaking

# VAD + RMS speaker detection (used when SPEAKING_BACKEND = "vad_rms")
VAD_THRESHOLD = 0.35  # Silero VAD probability above which = speech
VAD_RMS_BOUNDARY = 0.035  # initial adaptive boundary seed
VAD_RMS_EWMA_ALPHA = 0.3  # EWMA smoothing for adaptive RMS means
VAD_RMS_NOISE_FLOOR = 0.0  # initial RMS floor before room noise calibration
VAD_RMS_NOISE_FLOOR_ALPHA = 0.05  # slow EWMA for background noise during non-speech
VAD_RMS_SEED_HIGH_MULT = 2.0  # mean_high seed = boundary * this
VAD_RMS_SEED_LOW_MULT = 0.5  # mean_low seed = boundary * this

# Temporal smoothing (#9)
TEMPORAL_SMOOTHING_FRAMES = (
    7  # identity history window length for majority-vote smoothing
)
FACE_MAX_MOVE_PX = 100  # max bounding-box centre movement (px) to count as same face

# Camera
CAMERA_SOURCE = 0  # default webcam
CAMERA_WIDTH = 1536
CAMERA_HEIGHT = 2048
CAMERA_FPS = 30  # this is the actual framerate i am getting on my mac
ANDROID_CAMERA_FPS = 10  # Android camera streaming target fps

# Microphone
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds per audio chunk
VAD_AGGRESSIVENESS = 2  # webrtcvad aggressiveness (0-3)
SIMULATION_AUDIO_GAIN = 1.5  # boost weak glasses mic audio for simulation

# Audio processing
WHISPER_MODEL = "small"
WHISPER_LANGUAGE = "en"
SILENCE_THRESHOLD = 2.0  # seconds of silence before processing

# Live pipeline
LIVE_BUFFER_SECONDS = 10  # process audio/video in N-second windows
VISION_STRIDE = 5  # process faces every Nth frame for accelerated video processing

# Display
BBOX_COLOR = (0, 255, 0)  # green
UNKNOWN_BBOX_COLOR = (0, 0, 255)  # red
TEXT_COLOR = (255, 255, 255)  # white
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Companion app
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# Knowledge graph (Zep Graphiti → Neo4j)
SAVE_TO_MEMORY = os.getenv("SAVE_TO_MEMORY", "false").lower() == "true"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ravengers")

# Firebase authorization for API
# Download from: Firebase Console > Project Settings > Service Accounts > Generate new private key
FIREBASE_CREDENTIALS = os.getenv(
    "FIREBASE_CREDENTIALS"
)  # path to service account JSON file
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)
