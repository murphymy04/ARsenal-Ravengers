"""Audio input sources for the pipeline.

Microphone: captures live audio from hardware via sounddevice.
SimulatedMic: feeds pre-extracted audio at video frame rate for offline testing.

Both share the same interface:
    open(), advance_frame(), get_buffer_and_clear(), close()
"""

import threading

import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE


class Microphone:
    def __init__(self, sample_rate: int = SAMPLE_RATE, device: int | None = None):
        self.sample_rate = sample_rate
        self._device = device
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._drip_chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def open(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=512,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        mono = indata[:, 0].copy()
        with self._lock:
            self._chunks.append(mono)
            self._drip_chunks.append(mono)

    def advance_frame(self) -> np.ndarray:
        """Return all audio accumulated since last call (for per-frame dripping)."""
        with self._lock:
            if not self._drip_chunks:
                return np.array([], dtype=np.float32)
            buf = np.concatenate(self._drip_chunks)
            self._drip_chunks.clear()
            return buf

    def get_buffer_and_clear(self) -> np.ndarray:
        """Return all audio accumulated since last call (for transcription windows)."""
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


class SimulatedMic:
    """Feeds pre-extracted audio at video frame rate for offline testing."""

    def __init__(
        self,
        audio: np.ndarray,
        fps: float,
        sample_rate: int = SAMPLE_RATE,
        gain: float = 1.0,
        denoise: bool = False,
    ):
        self.sample_rate = sample_rate
        if denoise:
            import noisereduce as nr

            audio = nr.reduce_noise(y=audio, sr=sample_rate)
        self.audio = np.clip(audio * gain, -1.0, 1.0) if gain != 1.0 else audio
        self.samples_per_frame = sample_rate / fps
        self.cursor = 0.0
        self.window_chunks: list[np.ndarray] = []

    def advance_frame(self) -> np.ndarray:
        """Return audio samples for one video frame and advance cursor."""
        start = int(self.cursor)
        self.cursor += self.samples_per_frame
        end = int(self.cursor)
        end = min(end, len(self.audio))
        chunk = self.audio[start:end]
        self.window_chunks.append(chunk)
        return chunk

    def get_buffer_and_clear(self) -> np.ndarray:
        """Return all audio accumulated since last call (for transcription windows)."""
        if not self.window_chunks:
            return np.array([], dtype=np.float32)
        buf = np.concatenate(self.window_chunks)
        self.window_chunks.clear()
        return buf

    def open(self):
        pass

    def close(self):
        pass
