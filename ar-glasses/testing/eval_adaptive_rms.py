"""Evaluate adaptive RMS threshold across all test videos.

For each video:
1. Extracts raw RMS + VAD data (single pass).
2. Sweeps static boundaries and scores each with Fisher's discriminant.
3. Runs the adaptive two-mean tracker and records the final boundary.
4. Prints a comparison table.

Usage:
    python eval_adaptive_rms.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from config import (
    SIMULATION_AUDIO_GAIN,
    VAD_RMS_BOUNDARY,
    VAD_RMS_EWMA_ALPHA,
    VAD_RMS_SEED_HIGH_MULT,
    VAD_RMS_SEED_LOW_MULT,
    VAD_THRESHOLD,
)
from input.microphone import SimulatedMic
from pipeline.live import extract_audio_pcm, get_video_fps

TEST_VIDEOS_DIR = Path(__file__).parent.parent / "test_videos"
SAMPLE_RATE = 16000
SWEEP_MIN = 0.005
SWEEP_MAX = 0.400
SWEEP_STEP = 0.005


def extract_vad_rms(audio: np.ndarray, fps: float, model) -> pd.DataFrame:
    """Single VAD pass — returns per-frame (rms, vad_active) without any thresholding."""
    mic = SimulatedMic(audio, fps, gain=SIMULATION_AUDIO_GAIN, denoise=False)
    total_frames = int(len(mic.audio) / mic.samples_per_frame)

    audio_acc = np.array([], dtype=np.float32)
    rows = []

    for frame_idx in range(total_frames):
        samples = mic.advance_frame()
        if len(samples) == 0:
            continue

        audio_acc = np.concatenate([audio_acc, samples])

        vad_probs = []
        rms_values = []
        while len(audio_acc) >= 512:
            chunk = audio_acc[:512]
            audio_acc = audio_acc[512:]
            prob = model(torch.from_numpy(chunk), SAMPLE_RATE).item()
            vad_probs.append(prob)
            rms_values.append(float(np.sqrt(np.mean(chunk**2))))

        vad_max = max(vad_probs) if vad_probs else 0.0
        rms_mean = float(np.mean(rms_values)) if rms_values else 0.0

        rows.append(
            {
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "vad_prob": vad_max,
                "rms": rms_mean,
                "vad_active": vad_max > VAD_THRESHOLD,
            }
        )

    model.reset_states()
    return pd.DataFrame(rows)


def classify_static(df: pd.DataFrame, boundary: float) -> pd.Series:
    """Classify each frame given a fixed boundary (only meaningful where VAD is active)."""
    result = pd.Series("silence", index=df.index)
    speech_mask = df["vad_active"]
    result[speech_mask & (df["rms"] >= boundary)] = "wearer"
    result[speech_mask & (df["rms"] < boundary)] = "other"
    return result


def run_adaptive(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Run the two-mean adaptive tracker over pre-extracted VAD/RMS data.

    Returns (classifications, boundary_per_frame).
    """
    rms_mean_high = VAD_RMS_BOUNDARY * VAD_RMS_SEED_HIGH_MULT
    rms_mean_low = VAD_RMS_BOUNDARY * VAD_RMS_SEED_LOW_MULT
    adaptive_boundary = VAD_RMS_BOUNDARY

    classifications = []
    boundaries = []

    for _, row in df.iterrows():
        if row["vad_active"]:
            is_wearer = row["rms"] >= adaptive_boundary
            if is_wearer:
                rms_mean_high += VAD_RMS_EWMA_ALPHA * (row["rms"] - rms_mean_high)
            else:
                rms_mean_low += VAD_RMS_EWMA_ALPHA * (row["rms"] - rms_mean_low)
            adaptive_boundary = (rms_mean_high + rms_mean_low) / 2.0
            classifications.append("wearer" if is_wearer else "other")
        else:
            classifications.append("silence")
        boundaries.append(adaptive_boundary)

    return pd.Series(classifications, index=df.index), pd.Series(
        boundaries, index=df.index
    )


def fisher_score(rms: pd.Series, classification: pd.Series) -> float:
    """Fisher's discriminant ratio: (mean_high - mean_low)² / (var_high + var_low).

    Measures gap between group means relative to their spread.
    Does not penalize imbalanced group sizes.
    """
    speech_mask = classification != "silence"
    if speech_mask.sum() < 10:
        return 0.0

    wearer_rms = rms[classification == "wearer"]
    other_rms = rms[classification == "other"]

    if len(wearer_rms) < 3 or len(other_rms) < 3:
        return 0.0

    within_var = wearer_rms.var() + other_rms.var()
    if within_var < 1e-12:
        return 0.0

    return float((wearer_rms.mean() - other_rms.mean()) ** 2 / within_var)


def evaluate_video(video_path: Path, model) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  {video_path.stem}")
    print(f"{'=' * 60}")

    audio = extract_audio_pcm(video_path)
    fps = get_video_fps(video_path)

    print("  Extracting VAD + RMS...", end=" ", flush=True)
    raw_df = extract_vad_rms(audio, fps, model)
    speech_frames = raw_df["vad_active"].sum()
    print(f"{len(raw_df)} frames, {speech_frames} with speech")

    print("  Running adaptive...", end=" ", flush=True)
    adaptive_cls, adaptive_bounds = run_adaptive(raw_df)
    adaptive_final = adaptive_bounds.iloc[-1]
    adaptive_score = fisher_score(raw_df["rms"], adaptive_cls)
    print(f"boundary={adaptive_final:.4f}, score={adaptive_score:.4f}")

    boundaries = np.arange(SWEEP_MIN, SWEEP_MAX + SWEEP_STEP / 2, SWEEP_STEP)
    best_boundary = 0.0
    best_score = -1.0

    print(f"  Sweeping {len(boundaries)} static boundaries...", end=" ", flush=True)
    for b in boundaries:
        cls = classify_static(raw_df, b)
        score = fisher_score(raw_df["rms"], cls)
        if score > best_score:
            best_score = score
            best_boundary = b

    print(f"best={best_boundary:.4f} (score={best_score:.4f})")

    return {
        "video": video_path.stem,
        "best_static": best_boundary,
        "static_score": best_score,
        "adaptive_final": adaptive_final,
        "adaptive_score": adaptive_score,
        "adaptive_range": f"{adaptive_bounds.min():.4f}-{adaptive_bounds.max():.4f}",
    }


def main():
    videos = sorted(TEST_VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {TEST_VIDEOS_DIR}")
        sys.exit(1)

    print(f"Found {len(videos)} test videos")
    print("Loading Silero VAD model...", flush=True)
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )

    results = [evaluate_video(v, model) for v in videos]

    print(f"\n\n{'=' * 80}")
    print("  RESULTS")
    print(f"{'=' * 80}")
    header = (
        f"{'Video':<20} | {'Best Static':>11} | {'Static Score':>12} | "
        f"{'Adaptive Final':>14} | {'Adapt Score':>11} | {'Adaptive Range'}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['video']:<20} | {r['best_static']:>11.4f} | {r['static_score']:>12.4f} | "
            f"{r['adaptive_final']:>14.4f} | {r['adaptive_score']:>11.4f} | {r['adaptive_range']}"
        )


if __name__ == "__main__":
    main()
