"""Configuration for the input subsystem: camera, microphone, glasses adapter."""

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


# Glasses stream adapter
GLASSES_PAIR_QUEUE_MAX = 60
GLASSES_SPIN_INTERVAL_SEC = 0.005
GLASSES_PREBUFFER_SECONDS = 5.0
GLASSES_MAX_STAGING_SECONDS = 10.0
GLASSES_DROP_LAG_SECONDS = 0.2
