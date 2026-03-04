"""Evaluation dataset capture tool.

Streams from the webcam and saves full frames to a per-person folder so
that evaluate.py can run the complete detection → embedding → matching
pipeline on them — exactly as it runs in production.

Usage
-----
    python eval/capture.py --name Alice
    python eval/capture.py --name Alice --count 30 --output eval/dataset

Controls
--------
    c       capture current frame (only saved if a face is visible)
    SPACE   capture current frame (same as c)
    q       quit
"""

import sys
import argparse
from pathlib import Path

# Reach the ar-glasses package root regardless of where the script is invoked.
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from config import CAMERA_SOURCE
from processing.face_detector import FaceDetector


def _next_index(folder: Path) -> int:
    """Return the next available image index in folder."""
    existing = sorted(folder.glob("*.jpg"))
    if not existing:
        return 1
    return int(existing[-1].stem) + 1


def capture(name: str, output_dir: Path, target_count: int, camera_source: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initialising camera and face detector…")
    detector = FaceDetector()
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("ERROR: cannot open camera.")
        return

    saved = 0
    flash = 0  # frames remaining to show capture flash

    print(f"\nCapturing dataset for '{name}' → {output_dir}")
    print(f"Press [c] or [SPACE] to capture | [q] to quit | target: {target_count} images\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)
        preview = frame.copy()

        # Draw face boxes and blur score
        for face in faces:
            b = face.bbox
            color = (0, 255, 0) if face.blur_score >= 60 else (0, 165, 255)
            cv2.rectangle(preview, (b.x1, b.y1), (b.x2, b.y2), color, 2)
            cv2.putText(
                preview, f"blur={face.blur_score:.0f}",
                (b.x1, b.y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        # Capture flash overlay
        if flash > 0:
            cv2.rectangle(preview, (0, 0), (preview.shape[1], preview.shape[0]), (255, 255, 255), 8)
            flash -= 1

        # Status bar
        status = f"  '{name}'  saved: {saved}/{target_count}  faces: {len(faces)}"
        cv2.putText(
            preview, status,
            (10, preview.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
        )
        cv2.imshow("Capture", preview)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("c"), ord(" ")):
            if not faces:
                print("  No face detected — move closer or adjust lighting.")
            else:
                idx = _next_index(output_dir)
                path = output_dir / f"{idx:04d}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                flash = 6
                print(f"  Saved {path.name}  ({saved}/{target_count})")
                if saved >= target_count:
                    print(f"\nDone — {saved} images saved to {output_dir}")
                    break
        elif key == ord("q"):
            print(f"\nQuit — {saved} images saved to {output_dir}")
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Capture evaluation dataset images.")
    parser.add_argument("--name", required=True, help="Person's name (used as folder name).")
    parser.add_argument(
        "--output", default="eval/dataset",
        help="Root dataset directory (default: eval/dataset).",
    )
    parser.add_argument(
        "--count", type=int, default=20,
        help="Number of images to capture (default: 20).",
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_SOURCE,
        help="Camera index (default: 0).",
    )
    args = parser.parse_args()

    # Resolve output relative to the ar-glasses root, not cwd
    root = Path(__file__).parent.parent
    output_dir = (root / args.output / args.name).resolve()

    capture(args.name, output_dir, args.count, args.camera)


if __name__ == "__main__":
    main()
