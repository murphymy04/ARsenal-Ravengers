"""Real-time audio-visual active speaker detection using Light-ASD (CVPR 2023).

Captures microphone audio in a background thread, maintains a rolling buffer
of face crops per tracked face, and runs Light-ASD inference every
LIGHT_ASD_INFERENCE_INTERVAL video frames to decide who is speaking.

Usage in the video loop::

    detector = SpeakingDetector()     # starts mic thread
    ...
    for frame_idx, frame in enumerate(camera.frames()):
        faces, track_ids = tracker_step(frame)
        for face, tid in zip(faces, track_ids):
            detector.add_crop(tid, face.crop)   # RGB 112×112
        detector.run_inference(frame_idx)
        for face, tid in zip(faces, track_ids):
            face.is_speaking = detector.get_speaking(tid)
    ...
    detector.close()
"""

import collections
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Resolve MFCC backend once at import time — avoids repeated import attempts per call
try:
    from python_speech_features import mfcc as _psf_mfcc
except ImportError:
    _psf_mfcc = None

try:
    import librosa as _librosa
except ImportError:
    _librosa = None

if _psf_mfcc is None and _librosa is None:
    print("[SpeakingDetector] Neither python_speech_features nor librosa found.\n"
          "  Run: pip install python_speech_features")

from config import (
    LIGHT_ASD_WEIGHTS, LIGHT_ASD_VIDEO_FRAMES, LIGHT_ASD_INFERENCE_INTERVAL,
    LIGHT_ASD_MIN_FRAMES, LIGHT_ASD_SPEAKING_THRESHOLD,
    CAMERA_FPS, SAMPLE_RATE,
)


class SpeakingDetector:
    """Audio-visual active speaker detection, updating is_speaking per track."""

    def __init__(self, device: str = "cpu", use_mic: bool = True, fps: float = CAMERA_FPS):
        # Lazy import so the rest of the system works even if torch is absent
        from processing.light_asd.model import ASDInference
        self._model = ASDInference.load(Path(LIGHT_ASD_WEIGHTS), device=device)

        self._fps = fps
        self._sample_rate = SAMPLE_RATE

        # MFCC step: 1/(4*fps) seconds → ensures T_audio = 4 * T_visual
        self._mfcc_winstep = 1.0 / (4.0 * self._fps)
        self._mfcc_winlen = max(0.025, self._mfcc_winstep * 3)

        # Audio buffer — holds ~5 seconds of raw float32 mono samples
        max_samples = int(self._sample_rate * 6)
        self._audio_lock = threading.Lock()
        self._audio_buf: collections.deque = collections.deque(maxlen=max_samples)

        # Per-track buffers of grayscale face crops (uint8, 112×112)
        self._crop_bufs: dict[int, collections.deque] = {}

        # Full audio for offline/pre-loaded mode (bypasses the deque)
        self._full_audio: Optional[np.ndarray] = None

        # Latest speaking prediction per track
        self._speaking: dict[int, bool] = {}

        self._running = True
        self._mic_ok = False

        if use_mic:
            # Start microphone capture (daemon — dies with the main process)
            self._mic_thread = threading.Thread(target=self._mic_loop, daemon=True)
            self._mic_thread.start()
            # Wait briefly so we know if mic init succeeded
            time.sleep(0.3)
        else:
            self._mic_thread = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_audio(self, samples: np.ndarray):
        """Write pre-loaded audio samples into the buffer (bypasses mic).

        Args:
            samples: float32 mono audio at SAMPLE_RATE Hz.
        """
        self._full_audio = samples.copy()
        with self._audio_lock:
            self._audio_buf.extend(samples)
        self._mic_ok = True

    def add_crop(self, track_id: int, crop_rgb: np.ndarray):
        """Buffer a 112×112 RGB face crop for this track."""
        if track_id not in self._crop_bufs:
            self._crop_bufs[track_id] = collections.deque(maxlen=LIGHT_ASD_VIDEO_FRAMES)
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)  # (112, 112) uint8
        self._crop_bufs[track_id].append(gray)

    def run_inference(self, frame_count: int, active_track_ids: Optional[set] = None, timestamp: Optional[float] = None):
        """Run Light-ASD on all active tracks; call once per video frame.

        Args:
            frame_count: current frame index; inference runs every
                LIGHT_ASD_INFERENCE_INTERVAL frames.
            active_track_ids: set of track IDs still visible this frame.
                Buffers for tracks not in this set are evicted.
            timestamp: current video time in seconds. When provided, the
                audio window is anchored to this position instead of the
                tail of the buffer (needed for offline/pre-loaded audio).
        """
        if frame_count % LIGHT_ASD_INFERENCE_INTERVAL != 0:
            return
        if not self._mic_ok:
            return  # no audio → can't run AV model

        # Evict buffers for tracks that have disappeared
        if active_track_ids is not None:
            for tid in list(self._crop_bufs):
                if tid not in active_track_ids:
                    self.evict_track(tid)

        with self._audio_lock:
            audio_snapshot = np.array(self._audio_buf, dtype=np.float32)

        for tid, buf in list(self._crop_bufs.items()):
            if len(buf) < LIGHT_ASD_MIN_FRAMES:
                continue
            crops = list(buf)
            T = len(crops)

            # Extract MFCC for the corresponding audio window
            audio_sec = T / self._fps
            needed = int(audio_sec * self._sample_rate)
            if timestamp is not None and self._full_audio is not None:
                audio_end = int(timestamp * self._sample_rate)
                audio_start = max(0, audio_end - needed)
                if audio_end > len(self._full_audio) or audio_end - audio_start < needed // 2:
                    continue
                audio_window = self._full_audio[audio_start:audio_end]
            else:
                if len(audio_snapshot) < needed:
                    continue
                audio_window = audio_snapshot[-needed:]

            mfcc = _extract_mfcc(
                audio_window,
                sample_rate=self._sample_rate,
                winlen=self._mfcc_winlen,
                winstep=self._mfcc_winstep,
            )
            if mfcc is None:
                continue

            # Pad/trim to exactly T*4 audio frames
            T_audio = T * 4
            if len(mfcc) < T_audio:
                mfcc = np.pad(mfcc, ((0, T_audio - len(mfcc)), (0, 0)))
            else:
                mfcc = mfcc[:T_audio]

            visual = np.stack(crops, axis=0)   # (T, 112, 112) uint8
            prob = self._model.predict(visual, mfcc)
            self._speaking[tid] = prob >= LIGHT_ASD_SPEAKING_THRESHOLD
            print(f"\r[ASD] track={tid} speaking_prob={prob:.3f} {'SPEAKING' if self._speaking[tid] else '       '}", end="", flush=True)

    def get_speaking(self, track_id: int) -> bool:
        """Return the latest speaking prediction for a track."""
        return self._speaking.get(track_id, False)

    def evict_track(self, track_id: int):
        """Remove buffers for a track that has disappeared."""
        self._crop_bufs.pop(track_id, None)
        self._speaking.pop(track_id, None)

    def close(self):
        self._running = False

    # ------------------------------------------------------------------
    # Microphone capture (background thread)
    # ------------------------------------------------------------------

    def _mic_loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            print("[SpeakingDetector] sounddevice not installed — mic disabled. "
                  "Run: pip install sounddevice")
            return

        BLOCK = 512   # samples per callback (~32 ms at 16 kHz)

        def _callback(indata, frames, ts, status):
            # Copy mono channel before the callback returns (buffer may be reused)
            mono = indata[:, 0].copy()
            with self._audio_lock:
                self._audio_buf.extend(mono)

        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=BLOCK,
                callback=_callback,
            ):
                self._mic_ok = True
                print("[SpeakingDetector] Microphone active.")
                while self._running:
                    time.sleep(0.05)
        except Exception as e:
            print(f"[SpeakingDetector] Mic error: {e} — speaking detection disabled.")
            self._mic_ok = False


# ---------------------------------------------------------------------------
# MFCC helper
# ---------------------------------------------------------------------------

def _extract_mfcc(
    audio: np.ndarray,
    sample_rate: int,
    winlen: float,
    winstep: float,
    numcep: int = 13,
) -> Optional[np.ndarray]:
    """Return (T, 13) MFCC matrix, or None if no backend is available."""
    if _psf_mfcc is not None:
        return _psf_mfcc(audio, samplerate=sample_rate, numcep=numcep,
                         winlen=winlen, winstep=winstep).astype(np.float32)
    if _librosa is not None:
        hop = int(winstep * sample_rate)
        win = int(winlen * sample_rate)
        S = _librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=numcep,
            n_fft=win, hop_length=hop,
        )
        return S.T.astype(np.float32)
    return None
