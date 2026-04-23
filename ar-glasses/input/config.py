"""Configuration for the input subsystem: camera, microphone, glasses adapter."""

import os

from paths import DATA_DIR


# Camera
CAMERA_SOURCE = 0  # default webcam
CAMERA_WIDTH = 1536
CAMERA_HEIGHT = 2048
CAMERA_FPS = 30  # this is the actual framerate i am getting on my mac
ANDROID_CAMERA_FPS = 30  # Android camera streaming target fps

# Glasses USB stream resolution (scrcpy → FFmpeg output size, post-rotation)
GLASSES_VIDEO_WIDTH = 720
GLASSES_VIDEO_HEIGHT = 1280


# Microphone
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds per audio chunk
VAD_AGGRESSIVENESS = 2  # webrtcvad aggressiveness (0-3)
SIMULATION_AUDIO_GAIN = 1.5  # boost weak glasses mic audio for simulation


# Glasses stream adapter
GLASSES_PAIR_QUEUE_MAX = 60
GLASSES_SPIN_INTERVAL_SEC = 0.005
GLASSES_PREBUFFER_SECONDS = 0.05
GLASSES_MAX_STAGING_SECONDS = 2.0
GLASSES_DROP_LAG_SECONDS = 0.15


# Training data capture (records emitted glasses pairs to disk)
TRAINING_DATA = os.getenv("TRAINING_DATA", "false").lower() == "true"
TRAINING_DATA_DIR = DATA_DIR / "training_data"
