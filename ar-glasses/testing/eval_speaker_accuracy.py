"""Evaluate speaker classification accuracy against ground-truth annotations.

Parses annotated transcripts, runs adaptive VAD+RMS on each video,
and reports per-segment and overall speaker classification error rate.

Usage:
    python eval_speaker_accuracy.py
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config import (
    SIMULATION_AUDIO_GAIN,
    VAD_RMS_BOUNDARY,
    VAD_RMS_EWMA_ALPHA,
    VAD_RMS_SEED_HIGH_MULT,
    VAD_RMS_SEED_LOW_MULT,
    VAD_THRESHOLD,
)
import pandas as pd

from eval_adaptive_rms import extract_vad_rms
from pipeline.live import extract_audio_pcm, get_video_fps
from processing.vad_speaker import (
    create_adaptive_rms_state,
    update_noise_floor,
    update_speech_boundary,
)

TEST_VIDEOS_DIR = Path(__file__).parent.parent / "test_videos"
ANNOTATIONS_FILE = Path(__file__).parent.parent / "eval" / "transcripts_to_annotate.txt"

SEGMENT_RE = re.compile(r"\[\s*([\d.]+)\s*-\s*([\d.]+)\s*\]\s*(wearer|other)\s*:")


@dataclass
class GroundTruthSegment:
    start: float
    end: float
    speaker: str


def parse_annotations(path: Path) -> dict[str, list[GroundTruthSegment]]:
    annotations: dict[str, list[GroundTruthSegment]] = {}
    current_video = None

    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("=== ") and line.endswith(" ==="):
            current_video = line.strip("= ")
            annotations[current_video] = []
            continue

        match = SEGMENT_RE.match(line)
        if match and current_video:
            annotations[current_video].append(
                GroundTruthSegment(
                    start=float(match.group(1)),
                    end=float(match.group(2)),
                    speaker=match.group(3),
                )
            )

    return annotations


def run_adaptive_with_params(
    df,
    seed: float,
    alpha: float,
):
    adaptive_state = create_adaptive_rms_state(seed)

    classifications = []
    for _, row in df.iterrows():
        if row["vad_active"]:
            is_wearer, _ = update_speech_boundary(adaptive_state, row["rms"], alpha)
            classifications.append("wearer" if is_wearer else "other")
            continue

        update_noise_floor(adaptive_state, row["rms"])
        classifications.append("silence")

    return classifications, adaptive_state.boundary


def score_against_ground_truth(
    df,
    classifications: list[str],
    ground_truth: list[GroundTruthSegment],
) -> tuple[float, int, int, list[dict]]:
    """Returns (error_rate, errors, total, per_segment_details)."""
    total = 0
    errors = 0
    details = []

    for gt in ground_truth:
        mask = (df["timestamp"] >= gt.start) & (df["timestamp"] < gt.end)
        segment_frames = df[mask]

        if len(segment_frames) == 0:
            continue

        segment_cls = [classifications[i] for i in segment_frames.index]
        speech_cls = [c for c in segment_cls if c != "silence"]

        if not speech_cls:
            continue

        from collections import Counter

        counts = Counter(speech_cls)
        majority = counts.most_common(1)[0][0]
        segment_correct = majority == gt.speaker

        total += 1
        if not segment_correct:
            errors += 1

        details.append(
            {
                "start": gt.start,
                "end": gt.end,
                "expected": gt.speaker,
                "predicted": majority,
                "correct": segment_correct,
                "speech_frames": len(speech_cls),
                "frame_breakdown": dict(counts),
            }
        )

    error_rate = errors / total if total > 0 else 0.0
    return error_rate, errors, total, details


def evaluate_video(
    video_name: str,
    ground_truth: list[GroundTruthSegment],
    model,
    seed: float,
    alpha: float,
) -> tuple[float, int, int, list[dict], float]:
    video_path = TEST_VIDEOS_DIR / f"{video_name}.mp4"
    audio = extract_audio_pcm(video_path)
    fps = get_video_fps(video_path)

    df = extract_vad_rms(audio, fps, model)
    classifications, final_boundary = run_adaptive_with_params(df, seed, alpha)
    error_rate, errors, total, details = score_against_ground_truth(
        df, classifications, ground_truth
    )
    return error_rate, errors, total, details, final_boundary


def run_eval(
    annotations: dict[str, list[GroundTruthSegment]],
    model,
    seed: float = VAD_RMS_BOUNDARY,
    alpha: float = VAD_RMS_EWMA_ALPHA,
    verbose: bool = True,
) -> float:
    total_errors = 0
    total_segments = 0

    for video_name, ground_truth in annotations.items():
        error_rate, errors, total, details, final_boundary = evaluate_video(
            video_name, ground_truth, model, seed, alpha
        )
        total_errors += errors
        total_segments += total

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  {video_name}  (boundary={final_boundary:.4f})")
            print(f"{'=' * 60}")
            for d in details:
                status = "OK" if d["correct"] else "MISS"
                print(
                    f"  [{d['start']:6.2f}-{d['end']:6.2f}] "
                    f"{d['expected']:>6} -> {d['predicted']:>6}  "
                    f"{status}  ({d['speech_frames']} frames, {d['frame_breakdown']})"
                )
            print(f"  Error: {errors}/{total} = {error_rate:.1%}")

    overall_error = total_errors / total_segments if total_segments > 0 else 0.0

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  OVERALL: {total_errors}/{total_segments} = {overall_error:.1%} error")
        print(f"{'=' * 60}")

    return overall_error


def extract_all_video_data(
    annotations: dict[str, list[GroundTruthSegment]], model
) -> dict:
    """Extract VAD+RMS data for all annotated videos (single pass each)."""
    video_data = {}
    for video_name in annotations:
        video_path = TEST_VIDEOS_DIR / f"{video_name}.mp4"
        audio = extract_audio_pcm(video_path)
        fps = get_video_fps(video_path)
        df = extract_vad_rms(audio, fps, model)
        video_data[video_name] = df
        print(f"  {video_name}: {len(df)} frames, {df['vad_active'].sum()} speech")
    return video_data


def run_eval_on_data(
    video_data: dict,
    annotations: dict[str, list[GroundTruthSegment]],
    classify_fn,
    label: str,
    verbose: bool = True,
) -> float:
    """Run evaluation using a custom classification function.

    classify_fn(df, video_name, ground_truth) -> list[str] of per-frame classifications.
    Also accepts classify_fn(df) for backwards compatibility.
    """
    total_errors = 0
    total_segments = 0

    for video_name, ground_truth in annotations.items():
        df = video_data[video_name]
        try:
            classifications = classify_fn(df, video_name, ground_truth)
        except TypeError:
            classifications = classify_fn(df)
        error_rate, errors, total, details = score_against_ground_truth(
            df, classifications, ground_truth
        )
        total_errors += errors
        total_segments += total

        if verbose:
            print(f"\n  {video_name}: {errors}/{total} = {error_rate:.1%}")
            for d in details:
                if not d["correct"]:
                    print(
                        f"    MISS [{d['start']:6.2f}-{d['end']:6.2f}] "
                        f"{d['expected']:>6} -> {d['predicted']:>6}  "
                        f"({d['frame_breakdown']})"
                    )

    overall_error = total_errors / total_segments if total_segments > 0 else 0.0
    print(f"  {label}: {total_errors}/{total_segments} = {overall_error:.1%}")
    return overall_error


def classify_adaptive(df, seed, alpha, high_mult, low_mult):
    cls, _ = run_adaptive_with_params(df, seed, alpha)
    return cls


def classify_two_pass(df, initial_seed, alpha, high_mult, low_mult):
    """First pass establishes boundary, second pass uses it as seed."""
    _, pass1_boundary = run_adaptive_with_params(df, initial_seed, alpha)
    cls, _ = run_adaptive_with_params(df, pass1_boundary, alpha)
    return cls


def classify_percentile_seed(df, alpha, high_mult, low_mult, warmup_frames=50):
    """Seed from median RMS of first N speech frames."""
    speech_rms = df.loc[df["vad_active"], "rms"]
    if len(speech_rms) < warmup_frames:
        seed = speech_rms.median() if len(speech_rms) > 0 else 0.035
    else:
        seed = speech_rms.iloc[:warmup_frames].median()
    cls, _ = run_adaptive_with_params(df, seed, alpha)
    return cls


def classify_batch_kmeans(df):
    """Offline k-means (k=2) on all speech RMS values."""
    speech_mask = df["vad_active"]
    speech_rms = df.loc[speech_mask, "rms"].values

    if len(speech_rms) < 10:
        return ["silence"] * len(df)

    low = np.percentile(speech_rms, 25)
    high = np.percentile(speech_rms, 75)

    for _ in range(50):
        low_members = speech_rms[speech_rms < (low + high) / 2]
        high_members = speech_rms[speech_rms >= (low + high) / 2]
        if len(low_members) == 0 or len(high_members) == 0:
            break
        new_low = low_members.mean()
        new_high = high_members.mean()
        if abs(new_low - low) < 1e-6 and abs(new_high - high) < 1e-6:
            break
        low, high = new_low, new_high

    boundary = (low + high) / 2.0
    classifications = []
    for _, row in df.iterrows():
        if row["vad_active"]:
            classifications.append("wearer" if row["rms"] >= boundary else "other")
        else:
            classifications.append("silence")
    return classifications


def classify_batch_seed_adaptive(df, alpha, high_mult, low_mult):
    """Batch k-means to find seed, then run adaptive from there."""
    speech_mask = df["vad_active"]
    speech_rms = df.loc[speech_mask, "rms"].values

    if len(speech_rms) < 10:
        seed = 0.035
    else:
        low = np.percentile(speech_rms, 25)
        high = np.percentile(speech_rms, 75)
        for _ in range(50):
            low_members = speech_rms[speech_rms < (low + high) / 2]
            high_members = speech_rms[speech_rms >= (low + high) / 2]
            if len(low_members) == 0 or len(high_members) == 0:
                break
            new_low = low_members.mean()
            new_high = high_members.mean()
            if abs(new_low - low) < 1e-6 and abs(new_high - high) < 1e-6:
                break
            low, high = new_low, new_high
        seed = (low + high) / 2.0

    cls, _ = run_adaptive_with_params(df, seed, alpha)
    return cls


def smooth_classifications(classifications: list[str], window: int) -> list[str]:
    """Sliding window majority vote over frame classifications."""
    from collections import Counter

    smoothed = []
    half = window // 2
    for i in range(len(classifications)):
        if classifications[i] == "silence":
            smoothed.append("silence")
            continue
        start = max(0, i - half)
        end = min(len(classifications), i + half + 1)
        speech_in_window = [c for c in classifications[start:end] if c != "silence"]
        if not speech_in_window:
            smoothed.append("silence")
            continue
        counts = Counter(speech_in_window)
        smoothed.append(counts.most_common(1)[0][0])
    return smoothed


def classify_adaptive_smoothed(df, seed, alpha, high_mult, low_mult, window):
    cls, _ = run_adaptive_with_params(df, seed, alpha)
    return smooth_classifications(cls, window)


def run_adaptive_biased(df, seed, alpha, high_mult, low_mult, bias):
    """boundary = mean_low + bias * (mean_high - mean_low) instead of midpoint."""
    rms_mean_high = seed * high_mult
    rms_mean_low = seed * low_mult
    boundary = rms_mean_low + bias * (rms_mean_high - rms_mean_low)

    classifications = []
    for _, row in df.iterrows():
        if row["vad_active"]:
            is_wearer = row["rms"] >= boundary
            if is_wearer:
                rms_mean_high += alpha * (row["rms"] - rms_mean_high)
            else:
                rms_mean_low += alpha * (row["rms"] - rms_mean_low)
            boundary = rms_mean_low + bias * (rms_mean_high - rms_mean_low)
            classifications.append("wearer" if is_wearer else "other")
        else:
            classifications.append("silence")

    return classifications, boundary


def extract_spectral_features(audio: np.ndarray, fps: float, model) -> pd.DataFrame:
    """Extract RMS + spectral features per frame."""
    from input.microphone import SimulatedMic

    mic = SimulatedMic(audio, fps, gain=SIMULATION_AUDIO_GAIN, denoise=False)
    total_frames = int(len(mic.audio) / mic.samples_per_frame)

    audio_acc = np.array([], dtype=np.float32)
    rows = []
    sample_rate = 16000

    for frame_idx in range(total_frames):
        samples = mic.advance_frame()
        if len(samples) == 0:
            continue

        audio_acc = np.concatenate([audio_acc, samples])

        vad_probs = []
        rms_values = []
        low_energy_values = []
        high_energy_values = []

        while len(audio_acc) >= 512:
            chunk = audio_acc[:512]
            audio_acc = audio_acc[512:]
            prob = model(torch.from_numpy(chunk), sample_rate).item()
            vad_probs.append(prob)
            rms_values.append(float(np.sqrt(np.mean(chunk**2))))

            fft = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(512, 1.0 / sample_rate)
            low_mask = freqs < 1000
            high_mask = freqs >= 1000
            low_e = float(np.sum(fft[low_mask] ** 2))
            high_e = float(np.sum(fft[high_mask] ** 2))
            low_energy_values.append(low_e)
            high_energy_values.append(high_e)

        vad_max = max(vad_probs) if vad_probs else 0.0
        rms_mean = float(np.mean(rms_values)) if rms_values else 0.0
        low_e_mean = float(np.mean(low_energy_values)) if low_energy_values else 0.0
        high_e_mean = float(np.mean(high_energy_values)) if high_energy_values else 0.0
        spectral_ratio = low_e_mean / (high_e_mean + 1e-10)

        rows.append(
            {
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "vad_prob": vad_max,
                "rms": rms_mean,
                "low_energy": low_e_mean,
                "high_energy": high_e_mean,
                "spectral_ratio": spectral_ratio,
                "vad_active": vad_max > VAD_THRESHOLD,
            }
        )

    model.reset_states()
    return pd.DataFrame(rows)


def classify_spectral_adaptive(df, seed, alpha, high_mult, low_mult, feature="rms"):
    """Adaptive two-mean tracker using arbitrary feature column."""
    feat = df[feature].values
    vad_active = df["vad_active"].values

    mean_high = seed * high_mult
    mean_low = seed * low_mult
    boundary = seed

    classifications = []
    for i in range(len(df)):
        if vad_active[i]:
            is_wearer = feat[i] >= boundary
            if is_wearer:
                mean_high += alpha * (feat[i] - mean_high)
            else:
                mean_low += alpha * (feat[i] - mean_low)
            boundary = (mean_high + mean_low) / 2.0
            classifications.append("wearer" if is_wearer else "other")
        else:
            classifications.append("silence")

    return classifications


def classify_combined_score(df, rms_weight, spectral_weight, alpha):
    """Combine RMS and spectral ratio into a single score, then adaptive classify."""
    speech_mask = df["vad_active"]
    if speech_mask.sum() == 0:
        return ["silence"] * len(df)

    rms_vals = df["rms"].values
    spec_vals = df["spectral_ratio"].values

    speech_rms = rms_vals[speech_mask]
    speech_spec = spec_vals[speech_mask]

    rms_std = speech_rms.std() if speech_rms.std() > 0 else 1.0
    spec_std = speech_spec.std() if speech_spec.std() > 0 else 1.0
    rms_mean = speech_rms.mean()
    spec_mean = speech_spec.mean()

    score = (
        rms_weight * (rms_vals - rms_mean) / rms_std
        + spectral_weight * (spec_vals - spec_mean) / spec_std
    )

    seed = float(np.median(score[speech_mask]))
    mean_high = seed + 0.5
    mean_low = seed - 0.5
    boundary = seed

    classifications = []
    for i in range(len(df)):
        if not df["vad_active"].iloc[i]:
            classifications.append("silence")
            continue
        is_wearer = score[i] >= boundary
        if is_wearer:
            mean_high += alpha * (score[i] - mean_high)
        else:
            mean_low += alpha * (score[i] - mean_low)
        boundary = (mean_high + mean_low) / 2.0
        classifications.append("wearer" if is_wearer else "other")

    return classifications


def classify_with_inertia(df, seed, alpha, high_mult, low_mult, inertia_frames):
    """Require sustained evidence to switch speaker label."""
    rms_mean_high = seed * high_mult
    rms_mean_low = seed * low_mult
    boundary = seed

    current_speaker = "wearer"
    streak = 0

    classifications = []
    for _, row in df.iterrows():
        if not row["vad_active"]:
            classifications.append("silence")
            streak = 0
            continue

        raw_label = "wearer" if row["rms"] >= boundary else "other"

        if raw_label == current_speaker:
            streak = 0
        else:
            streak += 1
            if streak >= inertia_frames:
                current_speaker = raw_label
                streak = 0

        if raw_label == "wearer":
            rms_mean_high += alpha * (row["rms"] - rms_mean_high)
        else:
            rms_mean_low += alpha * (row["rms"] - rms_mean_low)
        boundary = (rms_mean_high + rms_mean_low) / 2.0

        classifications.append(current_speaker)

    return classifications


def classify_smoothed_rms_adaptive(df, rms_window, seed, alpha, high_mult, low_mult):
    """Smooth RMS with sliding window before adaptive classification."""
    rms_vals = df["rms"].values.copy()

    smoothed_rms = np.convolve(rms_vals, np.ones(rms_window) / rms_window, mode="same")

    mean_high = seed * high_mult
    mean_low = seed * low_mult
    boundary = seed

    classifications = []
    for i, (_, row) in enumerate(df.iterrows()):
        if not row["vad_active"]:
            classifications.append("silence")
            continue

        r = smoothed_rms[i]
        is_wearer = r >= boundary
        if is_wearer:
            mean_high += alpha * (r - mean_high)
        else:
            mean_low += alpha * (r - mean_low)
        boundary = (mean_high + mean_low) / 2.0
        classifications.append("wearer" if is_wearer else "other")

    return classifications


def classify_segment_level(
    df,
    ground_truth: list,
    seed: float,
    alpha: float,
    high_mult: float,
    low_mult: float,
):
    """Classify per segment using mean RMS of all speech frames in that segment.

    This is a post-hoc approach — uses segment boundaries from transcription.
    Returns per-frame classifications (all frames in a segment get same label).
    """
    mean_high = seed * high_mult
    mean_low = seed * low_mult
    boundary = seed

    classifications = ["silence"] * len(df)

    for gt in ground_truth:
        mask = (
            df["vad_active"]
            & (df["timestamp"] >= gt.start)
            & (df["timestamp"] < gt.end)
        )
        segment_frames = df[mask]
        if len(segment_frames) == 0:
            continue

        seg_rms = segment_frames["rms"].mean()
        is_wearer = seg_rms >= boundary
        label = "wearer" if is_wearer else "other"

        if is_wearer:
            mean_high += alpha * (seg_rms - mean_high)
        else:
            mean_low += alpha * (seg_rms - mean_low)
        boundary = (mean_high + mean_low) / 2.0

        for idx in segment_frames.index:
            classifications[idx] = label

    return classifications


def classify_rms_smoothed_segment_reclassify(
    df,
    ground_truth: list,
    rms_window: int,
    seed: float,
    alpha: float,
):
    """Two-stage: adaptive on smoothed RMS frames, then reclassify each segment
    by segment-mean smoothed RMS vs the final boundary."""
    rms_vals = df["rms"].values.copy()
    smoothed = np.convolve(rms_vals, np.ones(rms_window) / rms_window, mode="same")

    mean_high = seed * 1.5
    mean_low = seed * 0.3
    boundary = seed

    for i in range(len(df)):
        if not df["vad_active"].iloc[i]:
            continue
        r = smoothed[i]
        if r >= boundary:
            mean_high += alpha * (r - mean_high)
        else:
            mean_low += alpha * (r - mean_low)
        boundary = (mean_high + mean_low) / 2.0

    classifications = ["silence"] * len(df)
    for gt in ground_truth:
        mask = (
            df["vad_active"]
            & (df["timestamp"] >= gt.start)
            & (df["timestamp"] < gt.end)
        )
        seg_frames = df[mask]
        if len(seg_frames) == 0:
            continue
        seg_mean = smoothed[seg_frames.index].mean()
        label = "wearer" if seg_mean >= boundary else "other"
        for idx in seg_frames.index:
            classifications[idx] = label

    return classifications


def main():
    annotations = parse_annotations(ANNOTATIONS_FILE)
    print(f"Loaded annotations for {len(annotations)} videos")
    for name, segs in annotations.items():
        print(f"  {name}: {len(segs)} segments")

    print("\nLoading Silero VAD model...", flush=True)
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )

    print("\nExtracting VAD+RMS+spectral data...", flush=True)
    video_data = {}
    for video_name in annotations:
        video_path = TEST_VIDEOS_DIR / f"{video_name}.mp4"
        audio = extract_audio_pcm(video_path)
        fps = get_video_fps(video_path)
        df = extract_spectral_features(audio, fps, model)
        video_data[video_name] = df
        print(f"  {video_name}: {len(df)} frames, {df['vad_active'].sum()} speech")

    print("\n--- Feature distributions per video ---")
    for name, df in video_data.items():
        speech = df.loc[df["vad_active"]]
        print(
            f"  {name}: RMS median={speech['rms'].median():.4f}, "
            f"spectral_ratio median={speech['spectral_ratio'].median():.2f}, "
            f"vad_prob median={speech['vad_prob'].median():.3f}"
        )

    print("\n--- Per-speaker RMS distributions (ground truth) ---")
    for name, gt_segs in annotations.items():
        df = video_data[name]
        wearer_rms = []
        other_rms = []
        for gt in gt_segs:
            mask = (
                df["vad_active"]
                & (df["timestamp"] >= gt.start)
                & (df["timestamp"] < gt.end)
            )
            vals = df.loc[mask, "rms"].values
            if gt.speaker == "wearer":
                wearer_rms.extend(vals)
            else:
                other_rms.extend(vals)
        wearer_rms = np.array(wearer_rms)
        other_rms = np.array(other_rms)
        print(f"\n  {name}:")
        print(
            f"    wearer: n={len(wearer_rms)}, "
            f"mean={wearer_rms.mean():.4f}, med={np.median(wearer_rms):.4f}, "
            f"p25={np.percentile(wearer_rms, 25):.4f}, p75={np.percentile(wearer_rms, 75):.4f}"
        )
        print(
            f"    other:  n={len(other_rms)}, "
            f"mean={other_rms.mean():.4f}, med={np.median(other_rms):.4f}, "
            f"p25={np.percentile(other_rms, 25):.4f}, p75={np.percentile(other_rms, 75):.4f}"
        )
        overlap_lo = max(wearer_rms.min(), other_rms.min())
        overlap_hi = min(wearer_rms.max(), other_rms.max())
        wearer_in_overlap = np.sum(
            (wearer_rms >= overlap_lo) & (wearer_rms <= overlap_hi)
        )
        other_in_overlap = np.sum((other_rms >= overlap_lo) & (other_rms <= overlap_hi))
        print(
            f"    overlap zone: [{overlap_lo:.4f}, {overlap_hi:.4f}] — "
            f"{wearer_in_overlap}/{len(wearer_rms)} wearer, "
            f"{other_in_overlap}/{len(other_rms)} other in overlap"
        )
        optimal_boundary = (wearer_rms.mean() + other_rms.mean()) / 2
        wearer_correct = np.sum(wearer_rms >= optimal_boundary)
        other_correct = np.sum(other_rms < optimal_boundary)
        frame_acc = (wearer_correct + other_correct) / (
            len(wearer_rms) + len(other_rms)
        )
        print(
            f"    optimal static boundary (mean of means)={optimal_boundary:.4f}, "
            f"frame accuracy={frame_acc:.1%}"
        )

    alpha = VAD_RMS_EWMA_ALPHA
    high_mult = VAD_RMS_SEED_HIGH_MULT
    low_mult = VAD_RMS_SEED_LOW_MULT

    print("\n--- 1. Baseline adaptive (seed=0.035) ---")
    run_eval_on_data(
        video_data,
        annotations,
        lambda df: classify_adaptive(df, 0.035, alpha, high_mult, low_mult),
        "Baseline",
    )

    print("\n--- 2. Two-pass adaptive ---")
    run_eval_on_data(
        video_data,
        annotations,
        lambda df: classify_two_pass(df, 0.035, alpha, high_mult, low_mult),
        "Two-pass",
    )

    print("\n--- 3. Percentile-seeded adaptive ---")
    run_eval_on_data(
        video_data,
        annotations,
        lambda df: classify_percentile_seed(df, alpha, high_mult, low_mult),
        "Percentile-seed",
    )

    print("\n--- 4. Batch k-means (offline oracle) ---")
    run_eval_on_data(
        video_data,
        annotations,
        classify_batch_kmeans,
        "Batch k-means",
    )

    print("\n--- 5. Batch k-means seed + adaptive ---")
    run_eval_on_data(
        video_data,
        annotations,
        lambda df: classify_batch_seed_adaptive(df, alpha, high_mult, low_mult),
        "Batch-seed adaptive",
    )

    print("\n--- 6. Smoothed-RMS adaptive sweep ---")
    best_smooth_rms = 1.0
    best_smooth_rms_combo = ""
    for rms_w in [5, 10, 15, 20, 30]:
        for seed_val in [0.02, 0.035, 0.05]:
            for a in [0.05, 0.1, 0.2]:
                err = run_eval_on_data(
                    video_data,
                    annotations,
                    lambda df, w=rms_w, s=seed_val, al=a: (
                        classify_smoothed_rms_adaptive(df, w, s, al, 1.5, 0.3)
                    ),
                    f"rms_w={rms_w} seed={seed_val} a={a}",
                    verbose=False,
                )
                if err < best_smooth_rms:
                    best_smooth_rms = err
                    best_smooth_rms_combo = f"rms_w={rms_w} seed={seed_val} a={a}"

    print(f"\n  Best smoothed-RMS: {best_smooth_rms_combo} -> {best_smooth_rms:.1%}")

    print("\n--- 7. Segment-level adaptive ---")
    best_seg_error = 1.0
    best_seg_combo = ""
    for seed_val in [0.01, 0.02, 0.035, 0.05, 0.08]:
        for a in [0.05, 0.1, 0.2, 0.3, 0.5]:
            for hm in [1.5, 2.0, 3.0]:
                for lm in [0.2, 0.3, 0.5]:
                    err = run_eval_on_data(
                        video_data,
                        annotations,
                        lambda df, vn, gt, s=seed_val, al=a, h=hm, low=lm: (
                            classify_segment_level(df, gt, s, al, h, low)
                        ),
                        f"seed={seed_val} a={a} h={hm} l={lm}",
                        verbose=False,
                    )
                    if err < best_seg_error:
                        best_seg_error = err
                        best_seg_combo = f"seed={seed_val} a={a} h={hm} l={lm}"

    print(f"\n  Best segment-level: {best_seg_combo} -> {best_seg_error:.1%}")

    print("\n--- 8. Smoothed RMS + segment reclassify ---")
    best_reclassify = 1.0
    best_reclassify_combo = ""
    for rms_w in [5, 10, 15, 20, 30]:
        for seed_val in [0.01, 0.02, 0.035, 0.05, 0.08]:
            for a in [0.05, 0.1, 0.2, 0.3]:
                err = run_eval_on_data(
                    video_data,
                    annotations,
                    lambda df, vn, gt, w=rms_w, s=seed_val, al=a: (
                        classify_rms_smoothed_segment_reclassify(df, gt, w, s, al)
                    ),
                    f"rms_w={rms_w} seed={seed_val} a={a}",
                    verbose=False,
                )
                if err < best_reclassify:
                    best_reclassify = err
                    best_reclassify_combo = f"rms_w={rms_w} seed={seed_val} a={a}"

    print(f"\n  Best reclassify: {best_reclassify_combo} -> {best_reclassify:.1%}")

    print("\n\n=== SUMMARY ===")
    all_results = [
        ("smoothed-RMS adaptive", best_smooth_rms_combo, best_smooth_rms),
        ("segment-level", best_seg_combo, best_seg_error),
        ("reclassify", best_reclassify_combo, best_reclassify),
    ]
    for label, combo, err in all_results:
        print(f"  {label}: {combo} -> {err:.1%}")

    label, combo, overall_best = min(all_results, key=lambda x: x[2])
    print(f"\n  WINNER: {label} ({combo}) -> {overall_best:.1%}")


if __name__ == "__main__":
    main()
