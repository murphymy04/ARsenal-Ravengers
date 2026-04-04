"""Evaluate adaptive RMS boundary accuracy against ground truth.

Runs VadSpeaker on extracted audio from a test video and compares
per-frame wearer/other classification against ground truth segments.
No face detection or video processing needed — audio only.

Usage:
    python eval_boundary.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SAMPLE_RATE, SIMULATION_AUDIO_GAIN

PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_ROOT / "test_videos" / "myles_and_will.mp4"
GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "data" / "simulation_cache" / "myles_and_will_ground_truth.json"
)


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


def run_vad(audio: np.ndarray, fps: float) -> list[dict]:
    from processing.vad_speaker import VadSpeaker

    speaker = VadSpeaker(fps=fps, debug=True)
    samples_per_frame = SAMPLE_RATE / fps
    cursor = 0.0
    frame_results = []

    num_frames = int(len(audio) / samples_per_frame)
    for frame_idx in range(num_frames):
        start = int(cursor)
        cursor += samples_per_frame
        end = min(int(cursor), len(audio))
        chunk = audio[start:end]

        speaker.drip_audio(chunk)

        if speaker._vad_active:
            classification = "wearer" if speaker._is_wearer else "other"
        else:
            classification = "silence"

        frame_results.append(
            {
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "classification": classification,
                "rms_excess": max(
                    0.0,
                    float(np.sqrt(np.mean(chunk**2)))
                    - speaker._adaptive_state.noise_floor,
                ),
                "boundary_excess": speaker._adaptive_state.boundary
                - speaker._adaptive_state.noise_floor,
            }
        )

    speaker.close()
    return frame_results


def evaluate(frame_results: list[dict], ground_truth: list[dict]) -> dict:
    gt_wearer_correct = 0
    gt_wearer_total = 0
    gt_wearer_as_other = 0
    gt_other_correct = 0
    gt_other_total = 0
    gt_other_as_wearer = 0
    gt_speech_detected = 0
    gt_speech_total = 0

    per_segment = []

    for seg in ground_truth:
        is_wearer = seg["speaker"] == "wearer"
        gt_label = "wearer" if is_wearer else "other"

        seg_frames = [
            f for f in frame_results if seg["start"] <= f["timestamp"] < seg["end"]
        ]

        speech_frames = [f for f in seg_frames if f["classification"] != "silence"]
        correct_frames = [f for f in seg_frames if f["classification"] == gt_label]

        n_total = len(seg_frames)
        n_speech = len(speech_frames)
        n_correct = len(correct_frames)

        if n_total == 0:
            continue

        gt_speech_total += n_total
        gt_speech_detected += n_speech

        seg_accuracy = n_correct / n_total if n_total > 0 else 0.0
        seg_speech_rate = n_speech / n_total if n_total > 0 else 0.0

        if is_wearer:
            gt_wearer_total += n_total
            gt_wearer_correct += n_correct
            gt_wearer_as_other += sum(
                1 for f in seg_frames if f["classification"] == "other"
            )
        else:
            gt_other_total += n_total
            gt_other_correct += n_correct
            gt_other_as_wearer += sum(
                1 for f in seg_frames if f["classification"] == "wearer"
            )

        per_segment.append(
            {
                "speaker": seg["speaker"],
                "text": seg["text"][:50],
                "start": seg["start"],
                "end": seg["end"],
                "frames": n_total,
                "speech_detected": n_speech,
                "correct": n_correct,
                "accuracy": seg_accuracy,
                "speech_rate": seg_speech_rate,
            }
        )

    wearer_acc = gt_wearer_correct / gt_wearer_total if gt_wearer_total else 0
    other_acc = gt_other_correct / gt_other_total if gt_other_total else 0
    overall_acc = (
        (gt_wearer_correct + gt_other_correct) / (gt_wearer_total + gt_other_total)
        if (gt_wearer_total + gt_other_total)
        else 0
    )
    speech_detection_rate = (
        gt_speech_detected / gt_speech_total if gt_speech_total else 0
    )

    return {
        "wearer_accuracy": wearer_acc,
        "other_accuracy": other_acc,
        "overall_accuracy": overall_acc,
        "speech_detection_rate": speech_detection_rate,
        "wearer_as_other_frames": gt_wearer_as_other,
        "other_as_wearer_frames": gt_other_as_wearer,
        "wearer_total_frames": gt_wearer_total,
        "other_total_frames": gt_other_total,
        "per_segment": per_segment,
    }


def print_results(results: dict):
    print("\n" + "=" * 70)
    print("BOUNDARY EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Overall accuracy:       {results['overall_accuracy']:.1%}")
    print(f"  Wearer accuracy:        {results['wearer_accuracy']:.1%}")
    print(f"  Other accuracy:         {results['other_accuracy']:.1%}")
    print(f"  Speech detection rate:  {results['speech_detection_rate']:.1%}")
    print(
        f"  Wearer misclassified as other: {results['wearer_as_other_frames']}/{results['wearer_total_frames']}"
    )
    print(
        f"  Other misclassified as wearer: {results['other_as_wearer_frames']}/{results['other_total_frames']}"
    )

    print("\n  Per-segment breakdown:")
    print(f"  {'Speaker':<12} {'Time':>12} {'Acc':>6} {'Speech':>7} {'Text'}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 6} {'-' * 7} {'-' * 40}")
    for seg in results["per_segment"]:
        time_range = f"{seg['start']:5.1f}-{seg['end']:5.1f}s"
        print(
            f"  {seg['speaker']:<12} {time_range:>12} "
            f"{seg['accuracy']:5.0%} {seg['speech_rate']:6.0%}  "
            f"{seg['text']}"
        )


def main():
    import cv2

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
    results = evaluate(frame_results, ground_truth)
    print_results(results)

    return results


if __name__ == "__main__":
    main()
