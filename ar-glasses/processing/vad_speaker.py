"""VAD + RMS speaker detection.

Uses Silero VAD for voice activity detection and RMS amplitude to
distinguish wearer (loud, close to mic) from other speakers (quieter).
In 1-on-1 conversations the face in frame is the other person:

    VAD active + low RMS  →  face in frame is speaking
    VAD active + high RMS →  wearer is speaking (not on camera)
    VAD inactive          →  nobody speaking

The EWMA tracks the wearer's voice level and only updates when the
current RMS is in the wearer's range, so quiet speakers don't pull
it down.
"""

import numpy as np
import torch

from config import (
    CAMERA_FPS, SAMPLE_RATE,
    VAD_THRESHOLD, VAD_RMS_EWMA_ALPHA, VAD_WEARER_RATIO,
)


class VadSpeaker:
    """Drop-in replacement for SpeakingDetector using Silero VAD + RMS."""

    def __init__(self, fps: float = CAMERA_FPS):
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._fps = fps
        self._sample_rate = SAMPLE_RATE

        self._audio_acc = np.array([], dtype=np.float32)

        self._rms_ewma = 0.0
        self._ewma_initialized = False

        self._vad_active = False
        self._is_wearer = True

        self._speaking: dict[int, bool] = {}

    def drip_audio(self, samples: np.ndarray):
        if len(samples) == 0:
            return

        self._audio_acc = np.concatenate([self._audio_acc, samples])

        vad_any = False
        rms_values = []

        while len(self._audio_acc) >= 512:
            chunk = self._audio_acc[:512]
            self._audio_acc = self._audio_acc[512:]

            prob = self._model(
                torch.from_numpy(chunk), self._sample_rate
            ).item()

            if prob > VAD_THRESHOLD:
                vad_any = True
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                rms_values.append(rms)

        self._vad_active = vad_any

        if rms_values:
            rms_mean = float(np.mean(rms_values))

            if not self._ewma_initialized:
                self._rms_ewma = rms_mean
                self._ewma_initialized = True
            else:
                if rms_mean > self._rms_ewma * 0.7:
                    self._rms_ewma = (
                        VAD_RMS_EWMA_ALPHA * rms_mean
                        + (1 - VAD_RMS_EWMA_ALPHA) * self._rms_ewma
                    )

            self._is_wearer = rms_mean >= self._rms_ewma * VAD_WEARER_RATIO
            print(
                f"\r[VAD] rms={rms_mean:.4f} ewma={self._rms_ewma:.4f} "
                f"{'WEARER' if self._is_wearer else 'OTHER '}",
                end="", flush=True,
            )

    def add_crop(self, track_id: int, crop_rgb: np.ndarray):
        pass

    def run_inference(self, frame_count: int, active_track_ids: set | None = None):
        if active_track_ids is not None:
            for tid in list(self._speaking):
                if tid not in active_track_ids:
                    self._speaking.pop(tid, None)

        if not active_track_ids:
            return

        for tid in active_track_ids:
            self._speaking[tid] = False

        if self._vad_active and not self._is_wearer:
            target = min(active_track_ids)
            self._speaking[target] = True

    def get_speaking(self, track_id: int) -> bool:
        return self._speaking.get(track_id, False)

    def evict_track(self, track_id: int):
        self._speaking.pop(track_id, None)

    def close(self):
        self._model.reset_states()
