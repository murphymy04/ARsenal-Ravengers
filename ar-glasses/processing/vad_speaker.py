"""VAD + RMS speaker detection.

Uses Silero VAD for voice activity detection and an adaptive RMS boundary
to distinguish wearer (loud, close to mic) from other speakers (quieter).
The boundary tracks a background noise floor during non-speech, then runs
the two-mean adaptive tracker on speech RMS above that floor.

    VAD active + RMS >= boundary  →  wearer is speaking (not on camera)
    VAD active + RMS <  boundary  →  face in frame is speaking
    VAD inactive                  →  nobody speaking
"""

import collections
import csv
from dataclasses import dataclass

import numpy as np
import torch
from config import (
    CAMERA_FPS,
    DATA_DIR,
    SAMPLE_RATE,
    VAD_RMS_BOUNDARY,
    VAD_RMS_GMM_MIN_SAMPLES,
    VAD_RMS_NOISE_FLOOR,
    VAD_RMS_NOISE_FLOOR_ALPHA,
    VAD_RMS_RESET_SILENCE_SECONDS,
    VAD_RMS_SEED_HIGH_MULT,
    VAD_RMS_SEED_LOW_MULT,
    VAD_RMS_SPEECH_BUFFER,
    VAD_THRESHOLD,
)
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

_DEBUG_CSV = DATA_DIR / "vad_debug.csv"


@dataclass
class AdaptiveRmsState:
    noise_floor: float
    speech_mean_high: float
    speech_mean_low: float
    boundary: float


def create_adaptive_rms_state(seed_boundary: float) -> AdaptiveRmsState:
    seed_excess = max(0.0, seed_boundary - VAD_RMS_NOISE_FLOOR)
    return AdaptiveRmsState(
        noise_floor=VAD_RMS_NOISE_FLOOR,
        speech_mean_high=seed_excess * VAD_RMS_SEED_HIGH_MULT,
        speech_mean_low=seed_excess * VAD_RMS_SEED_LOW_MULT,
        boundary=seed_boundary,
    )


def boundary_excess(state: AdaptiveRmsState) -> float:
    return (state.speech_mean_high + state.speech_mean_low) / 2.0


def update_noise_floor(state: AdaptiveRmsState, rms_value: float) -> None:
    state.noise_floor += VAD_RMS_NOISE_FLOOR_ALPHA * (rms_value - state.noise_floor)
    state.boundary = state.noise_floor + boundary_excess(state)


def update_speech_distribution(
    state: AdaptiveRmsState,
    speech_excess_values: collections.deque[float],
) -> None:
    if not speech_excess_values:
        return

    if len(speech_excess_values) < VAD_RMS_GMM_MIN_SAMPLES:
        state.boundary = state.noise_floor + boundary_excess(state)
        return

    values = np.array(speech_excess_values, dtype=np.float32).reshape(-1, 1)
    seed_means = np.array(
        [[state.speech_mean_low], [state.speech_mean_high]],
        dtype=np.float32,
    )
    seed_means.sort(axis=0)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        means_init=seed_means,
        random_state=0,
    )
    with threadpool_limits(limits=1):
        gmm.fit(values)

    means = np.sort(gmm.means_.ravel())
    state.speech_mean_low = float(means[0])
    state.speech_mean_high = float(means[1])
    state.boundary = state.noise_floor + boundary_excess(state)


class VadSpeaker:
    """Drop-in replacement for SpeakingDetector using Silero VAD + RMS."""

    def __init__(
        self,
        fps: float = CAMERA_FPS,
        debug: bool = True,
        static_boundary: float | None = None,
    ):
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
        self._non_speech_frames = 0

        self._speaking: dict[int, bool] = {}

        self._static_boundary = static_boundary
        seed = static_boundary if static_boundary is not None else VAD_RMS_BOUNDARY
        self._speech_excess_values: collections.deque[float] = collections.deque(
            maxlen=VAD_RMS_SPEECH_BUFFER
        )

        self._adaptive_state = create_adaptive_rms_state(seed)
        self._adaptive_boundary = self._adaptive_state.boundary
        self._reset_silence_frames = max(
            1,
            round(VAD_RMS_RESET_SILENCE_SECONDS * self._fps),
        )

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
                    "noise_floor",
                    "rms_excess",
                    "speech_mean_low",
                    "speech_mean_high",
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
        rms_excess = max(0.0, rms_mean - self._adaptive_state.noise_floor)

        if self._vad_active:
            self._non_speech_frames = 0
            if self._static_boundary is None:
                self._speech_excess_values.append(rms_excess)
                update_speech_distribution(
                    self._adaptive_state,
                    self._speech_excess_values,
                )
                self._adaptive_boundary = self._adaptive_state.boundary
                self._is_wearer = rms_excess >= boundary_excess(self._adaptive_state)
            else:
                self._is_wearer = rms_mean >= self._static_boundary
        else:
            self._is_wearer = True
            self._non_speech_frames += 1
            if self._static_boundary is None:
                update_noise_floor(self._adaptive_state, rms_mean)
                self._adaptive_boundary = self._adaptive_state.boundary
                if self._non_speech_frames >= self._reset_silence_frames:
                    self._speech_excess_values.clear()

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
                    f"{self._adaptive_state.noise_floor:.6f}",
                    f"{rms_excess:.6f}",
                    f"{self._adaptive_state.speech_mean_low:.6f}",
                    f"{self._adaptive_state.speech_mean_high:.6f}",
                    f"{self._adaptive_boundary:.6f}",
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
