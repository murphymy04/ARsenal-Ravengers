"""Microphone input using sounddevice.

Captures audio in a background thread for speech processing.
Call open() to start recording, get_buffer_and_clear() to drain
accumulated samples, and close() to stop.
"""

import threading

import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE

class Microphone:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def open(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=512,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._chunks.append(indata[:, 0].copy())

    def get_buffer_and_clear(self) -> np.ndarray:
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)
            buf = np.concatenate(self._chunks)
            self._chunks.clear()
            return buf

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
