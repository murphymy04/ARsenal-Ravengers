"""Configuration for the processing subsystem: face detection, embedding, matching,
clustering, and speaking-detection backends.
"""

from paths import DATA_DIR

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
    360  # frames between accumulating new embeddings for a recognized person
)
MAX_EMBEDDINGS_PER_PERSON = 20  # cap to prevent unbounded growth per person
EMBEDDING_DIVERSITY_THRESHOLD = (
    0.15  # min cosine distance required to store a new embedding (#8)
)
MIN_SIGHTINGS_TO_CLUSTER = 12  # observations needed before promoting to a real cluster
PENDING_CLUSTER_SIMILARITY = 0.5  # cosine similarity for grouping pending observations
PENDING_EXPIRY_FRAMES = 180  # frames of absence before discarding a pending observation
MERGE_SIMILARITY_THRESHOLD = (
    0.35  # similarity above which two clusters are suggested for merging
)


# Face quality (#1, #2)
FACE_MIN_SIZE = 20  # px — ignore detections smaller than this (width or height)
FACE_BLUR_THRESHOLD = 70.0  # Laplacian variance — lower for H.264 compressed video


# Speaking detection backend
# Set SPEAKING_BACKEND = "light_asd" to use Light-ASD (audio-visual, more accurate).
# Set SPEAKING_BACKEND = "mediapipe" to use the MediaPipe jawOpen blendshape.
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
VAD_RMS_WEARER_EXCESS = 0.1  # anchored wearer RMS excess above noise floor
VAD_RMS_OTHER_EXCESS_INIT = 0.02  # initial other speaker RMS excess estimate
VAD_RMS_OTHER_ALPHA = 0.05  # EWMA rate for tracking other speaker mean
VAD_RMS_NOISE_FLOOR = 0.0  # initial RMS floor before room noise calibration
VAD_RMS_NOISE_FLOOR_ALPHA = 0.05  # slow EWMA for background noise during non-speech
VAD_RMS_EXCESS_SMOOTHING = 0.12  # EWMA on RMS excess signal before classification


# Temporal smoothing (#9)
TEMPORAL_SMOOTHING_FRAMES = (
    7  # identity history window length for majority-vote smoothing
)
FACE_MAX_MOVE_PX = 100  # max bounding-box centre movement (px) to count as same face
