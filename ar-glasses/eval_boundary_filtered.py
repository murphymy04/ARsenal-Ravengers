"""Evaluate VAD+RMS boundary with minimum duration filtering.

Runs the standard eval_boundary pipeline, then post-processes the
frame-level classifications by merging short segments into their neighbors.
Compares accuracy across multiple min_duration values.

Usage:
    python eval_boundary_filtered.py [--min-frames 5]
"""

import argparse
import copy

import cv2

from eval_boundary import (
    VIDEO_PATH,
    evaluate,
    extract_audio,
    load_ground_truth,
    print_results,
    run_vad,
    GROUND_TRUTH_PATH,
)
from config import SAMPLE_RATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VAD+RMS with min duration filtering."
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="Default min segment length in frames.",
    )
    return parser.parse_args()


def filter_min_duration(frame_results: list[dict], min_frames: int) -> list[dict]:
    """Merge segments shorter than min_frames into their longer neighbor."""
    if min_frames <= 1:
        return frame_results

    filtered = copy.deepcopy(frame_results)

    changed = True
    while changed:
        changed = False
        runs = _build_runs(filtered)

        for i, (label, start_idx, length) in enumerate(runs):
            if length >= min_frames:
                continue

            left_len = runs[i - 1][2] if i > 0 else 0
            right_len = runs[i + 1][2] if i < len(runs) - 1 else 0

            if left_len == 0 and right_len == 0:
                continue

            if left_len >= right_len:
                new_label = runs[i - 1][0]
            else:
                new_label = runs[i + 1][0]

            for j in range(start_idx, start_idx + length):
                filtered[j]["classification"] = new_label

            changed = True
            break

    return filtered


def _build_runs(frame_results: list[dict]) -> list[tuple[str, int, int]]:
    """Group consecutive frames by classification label.

    Returns list of (label, start_index, length).
    """
    if not frame_results:
        return []

    runs = []
    current_label = frame_results[0]["classification"]
    start = 0

    for i in range(1, len(frame_results)):
        if frame_results[i]["classification"] != current_label:
            runs.append((current_label, start, i - start))
            current_label = frame_results[i]["classification"]
            start = i

    runs.append((current_label, start, len(frame_results) - start))
    return runs


def main():
    _args = parse_args()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    print(f"Video: {VIDEO_PATH.name} ({fps:.1f} fps)")
    print("Extracting audio...")
    audio = extract_audio(VIDEO_PATH)
    print(f"Audio: {len(audio)} samples ({len(audio) / SAMPLE_RATE:.1f}s)")

    print("Running VadSpeaker...")
    frame_results = run_vad(audio, fps)
    print(f"Processed {len(frame_results)} frames")

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    min_frame_values = [0, 3, 5, 7, 10, 15, 20, 30]

    all_results = []
    for min_f in min_frame_values:
        if min_f == 0:
            filtered = frame_results
        else:
            filtered = filter_min_duration(frame_results, min_f)

        result = evaluate(filtered, ground_truth)
        result["min_frames"] = min_f
        result["min_seconds"] = min_f / fps
        all_results.append(result)

    print("\n" + "=" * 70)
    print("MIN DURATION FILTERING COMPARISON")
    print("=" * 70)
    print(
        f"  {'min_frames':>10} {'min_sec':>8} {'overall':>8} "
        f"{'wearer':>8} {'other':>8} {'speech':>8} "
        f"{'w→o':>5} {'o→w':>5}"
    )
    print(
        f"  {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 5} {'-' * 5}"
    )

    for r in all_results:
        print(
            f"  {r['min_frames']:>10} {r['min_seconds']:>7.2f}s "
            f"{r['overall_accuracy']:>7.1%} "
            f"{r['wearer_accuracy']:>7.1%} "
            f"{r['other_accuracy']:>7.1%} "
            f"{r['speech_detection_rate']:>7.1%} "
            f"{r['wearer_as_other_frames']:>5} "
            f"{r['other_as_wearer_frames']:>5}"
        )

    best = max(all_results, key=lambda r: r["overall_accuracy"])
    print(
        f"\n  Best: min_frames={best['min_frames']} "
        f"({best['min_seconds']:.2f}s) → {best['overall_accuracy']:.1%} overall"
    )

    print("\n\nDetailed results for best min_frames:")
    print_results(best)


if __name__ == "__main__":
    main()
