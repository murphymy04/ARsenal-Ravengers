"""Configuration for the live pipeline: whisper transcription, retrieval,
knowledge graph, HUD broadcast, and save-to-memory behavior.
"""

import os

# Audio processing
WHISPER_MODEL = "small"
WHISPER_LANGUAGE = "en"
SILENCE_THRESHOLD = 2.0  # seconds of silence before processing


# Live pipeline
LIVE_BUFFER_SECONDS = 10  # process audio/video in N-second windows
VISION_STRIDE = 5  # process faces every Nth frame for accelerated video processing


# Knowledge graph (Zep Graphiti → Neo4j)
SAVE_TO_MEMORY = os.getenv("SAVE_TO_MEMORY", "false").lower() == "true"
RETRIEVAL_ENABLED = os.getenv("RETRIEVAL_ENABLED", "false").lower() == "true"
RETRIEVAL_COOLDOWN_SECONDS = float(os.getenv("RETRIEVAL_COOLDOWN_SECONDS", "30"))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ravengers")


# HUD broadcast (Unity glasses app connects over WebSocket)
HUD_BROADCAST_ENABLED = os.getenv("HUD_BROADCAST_ENABLED", "false").lower() == "true"
HUD_BROADCAST_HOST = os.getenv("HUD_BROADCAST_HOST", "0.0.0.0")
HUD_BROADCAST_PORT = int(os.getenv("HUD_BROADCAST_PORT", "8765"))
