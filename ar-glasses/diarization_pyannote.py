"""Evaluate pyannote speaker diarization against ground truth.

Runs pyannote/speaker-diarization-3.1 on extracted audio from a test video
and compares per-segment speaker assignment against ground truth labels.

Usage:
    python diarization_pyannote.py [--min-duration-on 0.3] [--min-duration-off 0.3]
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from pyannote.audio import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

SAMPLE_RATE = 16000
SIMULATION_AUDIO_GAIN = 1.5
VIDEO_PATH = PROJECT_ROOT / "test_videos" / "myles_and_will.mp4"
GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "data" / "simulation_cache" / "myles_and_will_ground_truth.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pyannote diarization evaluation.")
    parser.add_argument(
        "--min-duration-on",
        type=float,
        default=0.0,
        help="Minimum duration of speech turns (seconds).",
    )
    parser.add_argument(
        "--min-duration-off",
        type=float,
        default=0.0,
        help="Minimum duration of non-speech gaps (seconds).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Number of speakers to detect.",
    )
    return parser.parse_args()


def extract_audio(video_path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-loglevel",
        "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    raw = np.frombuffer(result.stdout, dtype=np.float32)
    return np.clip(raw * SIMULATION_AUDIO_GAIN, -1.0, 1.0)


def load_ground_truth(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["combined"]


def run_pyannote(
    audio: np.ndarray,
    num_speakers: int,
    min_duration_on: float,
    min_duration_off: float,
) -> list[dict]:
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN not set in environment or .env")

    print("Loading pyannote pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=token
    )

    if min_duration_on > 0 or min_duration_off > 0:
        params = pipeline.parameters(instantiated=True)
        if min_duration_on > 0:
            params["min_duration_on"] = min_duration_on
        if min_duration_off > 0:
            params["min_duration_off"] = min_duration_off
        pipeline.instantiate(params)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio, SAMPLE_RATE)

    try:
        print("Running diarization...")
        diarization = pipeline(tmp_path, num_speakers=num_speakers)
    finally:
        os.unlink(tmp_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    return segments


def evaluate(pyannote_segments: list[dict], ground_truth: list[dict]) -> dict:
    speaker_labels = sorted({s["speaker"] for s in pyannote_segments})
    if len(speaker_labels) < 2:
        print(f"Warning: pyannote found only {len(speaker_labels)} speaker(s)")
        speaker_labels = speaker_labels + ["NONE"] * (2 - len(speaker_labels))

    def overlap(seg_start, seg_end, query_start, query_end) -> float:
        return max(0.0, min(seg_end, query_end) - max(seg_start, query_start))

    best_mapping = None
    best_accuracy = -1.0

    from itertools import permutations

    for perm in permutations(speaker_labels[:2]):
        mapping = {"wearer": perm[0], "other": perm[1]}
        total_time = 0.0
        correct_time = 0.0

        for gt_seg in ground_truth:
            gt_label = "wearer" if gt_seg["speaker"] == "wearer" else "other"
            gt_duration = gt_seg["end"] - gt_seg["start"]
            total_time += gt_duration

            expected_pyannote_label = mapping[gt_label]
            matched_time = sum(
                overlap(ps["start"], ps["end"], gt_seg["start"], gt_seg["end"])
                for ps in pyannote_segments
                if ps["speaker"] == expected_pyannote_label
            )
            correct_time += matched_time

        acc = correct_time / total_time if total_time > 0 else 0
        if acc > best_accuracy:
            best_accuracy = acc
            best_mapping = mapping

    wearer_total = 0.0
    wearer_correct = 0.0
    other_total = 0.0
    other_correct = 0.0
    per_segment = []

    for gt_seg in ground_truth:
        gt_label = "wearer" if gt_seg["speaker"] == "wearer" else "other"
        gt_duration = gt_seg["end"] - gt_seg["start"]

        expected_label = best_mapping[gt_label]
        matched = sum(
            overlap(ps["start"], ps["end"], gt_seg["start"], gt_seg["end"])
            for ps in pyannote_segments
            if ps["speaker"] == expected_label
        )
        seg_acc = matched / gt_duration if gt_duration > 0 else 0

        if gt_label == "wearer":
            wearer_total += gt_duration
            wearer_correct += matched
        else:
            other_total += gt_duration
            other_correct += matched

        per_segment.append(
            {
                "speaker": gt_seg["speaker"],
                "text": gt_seg["text"][:50],
                "start": gt_seg["start"],
                "end": gt_seg["end"],
                "duration": gt_duration,
                "correct_time": matched,
                "accuracy": seg_acc,
            }
        )

    consistency = compute_consistency(pyannote_segments, ground_truth)

    return {
        "mapping": best_mapping,
        "overall_accuracy": best_accuracy,
        "wearer_accuracy": wearer_correct / wearer_total if wearer_total else 0,
        "other_accuracy": other_correct / other_total if other_total else 0,
        "wearer_total_seconds": wearer_total,
        "other_total_seconds": other_total,
        "consistency": consistency,
        "per_segment": per_segment,
    }


def compute_consistency(
    pyannote_segments: list[dict], ground_truth: list[dict]
) -> dict:
    """For each GT speaker class, what % of speech time goes to one pyannote label."""

    def overlap(s1, e1, s2, e2) -> float:
        return max(0.0, min(e1, e2) - max(s1, s2))

    result = {}
    for gt_class in ["wearer", "other"]:
        gt_segs = [
            s
            for s in ground_truth
            if (s["speaker"] == "wearer") == (gt_class == "wearer")
        ]
        label_times: dict[str, float] = {}
        total = 0.0
        for gt_seg in gt_segs:
            for ps in pyannote_segments:
                ov = overlap(ps["start"], ps["end"], gt_seg["start"], gt_seg["end"])
                if ov > 0:
                    label_times[ps["speaker"]] = label_times.get(ps["speaker"], 0) + ov
                    total += ov

        if total > 0:
            dominant_label = max(label_times, key=label_times.get)
            dominant_pct = label_times[dominant_label] / total
        else:
            dominant_label = "N/A"
            dominant_pct = 0.0

        result[gt_class] = {
            "dominant_label": dominant_label,
            "consistency": dominant_pct,
            "label_breakdown": {k: v / total for k, v in label_times.items()}
            if total > 0
            else {},
        }

    return result


def print_results(results: dict, pyannote_segments: list[dict]):
    print("\n" + "=" * 70)
    print("PYANNOTE DIARIZATION RESULTS")
    print("=" * 70)

    print(f"\n  Mapping: {results['mapping']}")
    print(f"  Overall accuracy:  {results['overall_accuracy']:.1%}")
    print(f"  Wearer accuracy:   {results['wearer_accuracy']:.1%}")
    print(f"  Other accuracy:    {results['other_accuracy']:.1%}")

    print("\n  Speaker consistency:")
    for cls, info in results["consistency"].items():
        print(
            f"    {cls}: {info['consistency']:.1%} assigned to {info['dominant_label']}"
        )
        for label, pct in info["label_breakdown"].items():
            print(f"      {label}: {pct:.1%}")

    print(f"\n  Pyannote segments: {len(pyannote_segments)}")
    print("\n  Per-segment breakdown:")
    print(f"  {'Speaker':<12} {'Time':>12} {'Acc':>6} {'Text'}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 6} {'-' * 40}")
    for seg in results["per_segment"]:
        time_range = f"{seg['start']:5.1f}-{seg['end']:5.1f}s"
        print(
            f"  {seg['speaker']:<12} {time_range:>12} "
            f"{seg['accuracy']:5.0%}  "
            f"{seg['text']}"
        )


def main():
    args = parse_args()
    print(f"Video: {VIDEO_PATH.name}")
    print(f"num_speakers={args.num_speakers}")
    if args.min_duration_on > 0:
        print(f"min_duration_on={args.min_duration_on}")
    if args.min_duration_off > 0:
        print(f"min_duration_off={args.min_duration_off}")

    print("Extracting audio...")
    audio = extract_audio(VIDEO_PATH)
    print(f"Audio: {len(audio)} samples ({len(audio) / SAMPLE_RATE:.1f}s)")

    pyannote_segments = run_pyannote(
        audio, args.num_speakers, args.min_duration_on, args.min_duration_off
    )

    print(f"\nPyannote found {len(pyannote_segments)} segments")
    for seg in pyannote_segments:
        print(f"  {seg['start']:6.1f} - {seg['end']:6.1f}  {seg['speaker']}")

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    results = evaluate(pyannote_segments, ground_truth)
    print_results(results, pyannote_segments)


if __name__ == "__main__":
    main()
