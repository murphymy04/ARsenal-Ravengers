"""Microphone input using PyAudio.

Captures audio chunks for speech processing.
Stubbed for Phase 2 implementation.
"""

import numpy as np
from typing import Generator

from config import SAMPLE_RATE, CHUNK_DURATION


class Microphone:
    """PyAudio microphone wrapper (stub - Phase 2)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_duration: float = CHUNK_DURATION):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)

    def open(self):
        """Open the microphone stream."""
        raise NotImplementedError("Microphone capture is Phase 2 - requires PyAudio")

    def read_chunk(self) -> np.ndarray:
        """Read a single chunk of float32 audio."""
        raise NotImplementedError("Microphone capture is Phase 2 - requires PyAudio")

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield float32 audio chunks continuously."""
        raise NotImplementedError("Microphone capture is Phase 2 - requires PyAudio")

    def close(self):
        """Close the microphone stream."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
