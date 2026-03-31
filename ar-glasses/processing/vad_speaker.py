"""VAD + RMS speaker detection.

Uses Silero VAD for voice activity detection and a fixed RMS boundary to
distinguish wearer (loud, close to mic) from other speakers (quieter).
In 1-on-1 conversations the face in frame is the other person:

    VAD active + RMS >= boundary  →  wearer is speaking (not on camera)
    VAD active + RMS <  boundary  →  face in frame is speaking
    VAD inactive                  →  nobody speaking
"""

import csv

import numpy as np
import torch
from config import (
    CAMERA_FPS,
    DATA_DIR,
    SAMPLE_RATE,
    VAD_RMS_BOUNDARY,
    VAD_THRESHOLD,
)

_DEBUG_CSV = DATA_DIR / "vad_debug.csv"


class VadSpeaker:
    """Drop-in replacement for SpeakingDetector using Silero VAD + RMS."""

    def __init__(self, fps: float = CAMERA_FPS, debug: bool = True):
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._fps = fps
        self._sample_rate = SAMPLE_RATE

        self._audio_acc = np.array([], dtype=np.float32)

        self._vad_active = False
        self._is_wearer = True
        self._frame_idx = 0

        self._speaking: dict[int, bool] = {}

        self._debug = debug
        self._debug_file = None
        self._debug_writer = None
        if debug:
            self._debug_file = open(_DEBUG_CSV, "w", newline="")  # noqa: SIM115
            self._debug_writer = csv.writer(self._debug_file)
            self._debug_writer.writerow(
                [
                    "frame",
                    "timestamp",
                    "vad_prob",
                    "rms",
                    "boundary",
                    "is_wearer",
                    "classification",
                ]
            )

    def drip_audio(self, samples: np.ndarray):
        if len(samples) == 0:
            self._frame_idx += 1
            return

        self._audio_acc = np.concatenate([self._audio_acc, samples])

        vad_probs = []
        rms_values = []

        while len(self._audio_acc) >= 512:
            chunk = self._audio_acc[:512]
            self._audio_acc = self._audio_acc[512:]

            prob = self._model(torch.from_numpy(chunk), self._sample_rate).item()

            vad_probs.append(prob)
            rms = float(np.sqrt(np.mean(chunk**2)))
            rms_values.append(rms)

        vad_max = max(vad_probs) if vad_probs else 0.0
        self._vad_active = vad_max > VAD_THRESHOLD
        rms_mean = float(np.mean(rms_values)) if rms_values else 0.0

        if self._vad_active:
            self._is_wearer = rms_mean >= VAD_RMS_BOUNDARY
        else:
            self._is_wearer = True

        if self._vad_active:
            classification = "wearer" if self._is_wearer else "other"
        else:
            classification = "silence"

        if self._debug and self._debug_writer:
            timestamp = self._frame_idx / self._fps
            self._debug_writer.writerow(
                [
                    self._frame_idx,
                    f"{timestamp:.3f}",
                    f"{vad_max:.4f}",
                    f"{rms_mean:.6f}",
                    f"{VAD_RMS_BOUNDARY:.6f}",
                    int(self._is_wearer),
                    classification,
                ]
            )

        self._frame_idx += 1

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
        if self._debug_file:
            self._debug_file.close()
            self._debug_file = None
            print(f"\n[VAD] Debug CSV written to {_DEBUG_CSV}")
