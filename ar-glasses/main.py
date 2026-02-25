"""AR Glasses Prototype - Main Orchestrator.

Wires together the face recognition pipeline:
  Camera -> Detect -> Embed -> Match -> Display

Two threads + optional third:
  Thread 1 (main): Video loop - face recognition + display
  Thread 2: Audio loop (Phase 2 - stubbed)
  Thread 3: Companion app (Phase 2 - stubbed)
"""

import sys
import time
import argparse

from config import CAMERA_SOURCE
from input.camera import Camera
from processing.face_detector import FaceDetector
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from storage.database import Database
from storage.enrollment import Enrollment
from output.display import Display


def run_video_loop(
    camera: Camera,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    matcher: FaceMatcher,
    display: Display,
    db: Database,
):
    """Main video loop: detect, embed, match, display."""
    print("Starting video loop... Press 'q' to quit.")

    # Load known people into the matcher
    people = db.get_all_people()
    matcher.update_gallery(people)
    print(f"Loaded {len(people)} known people from database.")

    frame_count = 0
    start_time = time.time()

    for frame in camera.frames():
        timestamp = time.time()

        # Detect faces
        faces = detector.detect(frame, timestamp=timestamp)

        # Embed and match each face
        matches = []
        for face in faces:
            embedding = embedder.embed(face.crop)
            match = matcher.match(embedding)
            matches.append(match)

            # Update last_seen for recognized people
            if match.is_known:
                db.update_last_seen(match.person_id)

        # Draw overlays and display
        display.draw(frame, faces, matches)
        if not display.show(frame):
            break

        # FPS counter (every 30 frames)
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"\rFPS: {fps:.1f} | Faces: {len(faces)}", end="", flush=True)

    print("\nVideo loop ended.")


def enroll_mode(db: Database, detector: FaceDetector, embedder: FaceEmbedder):
    """Interactive enrollment from webcam."""
    import cv2

    enrollment = Enrollment(db, detector, embedder)

    name = input("Enter person's name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    print("Opening camera... (if this hangs, run from Windows PowerShell instead)")
    camera = Camera()
    print("Enrollment mode. Press 'c' to capture, 'q' to quit.")

    captured_images = []
    for frame in camera.frames():
        faces = detector.detect(frame)
        # Draw detection boxes for preview
        for face in faces:
            b = face.bbox
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)

        cv2.putText(
            frame, f"Enrolling: {name} | Captured: {len(captured_images)} | 'c'=capture 'q'=done",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        cv2.imshow("Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and faces:
            captured_images.append(frame.copy())
            print(f"  Captured image {len(captured_images)}")
        elif key == ord("q"):
            break

    camera.close()
    cv2.destroyAllWindows()

    if captured_images:
        person_id = enrollment.enroll_from_images(name, captured_images)
        if person_id:
            print(f"Enrolled '{name}' with ID {person_id} ({len(captured_images)} images)")
        else:
            print("Failed to detect a face in any captured image.")
    else:
        print("No images captured.")


def main():
    parser = argparse.ArgumentParser(description="AR Glasses Prototype")
    parser.add_argument(
        "--mode", choices=["run", "enroll"], default="run",
        help="'run' for live recognition, 'enroll' to add a new person",
    )
    parser.add_argument(
        "--camera", type=int, default=CAMERA_SOURCE,
        help="Camera source index (default: 0)",
    )
    args = parser.parse_args()

    # Initialize shared components
    print("Initializing database...")
    db = Database()
    print("Initializing face detector (MediaPipe)...")
    detector = FaceDetector()
    print("Loading EdgeFace model (this may take 10-20s)...")
    embedder = FaceEmbedder()
    print("Ready.")
    matcher = FaceMatcher()

    try:
        if args.mode == "enroll":
            enroll_mode(db, detector, embedder)
        else:
            camera = Camera(source=args.camera)
            display = Display()
            try:
                run_video_loop(camera, detector, embedder, matcher, display, db)
            finally:
                camera.close()
                display.close()
    finally:
        detector.close()
        db.close()


if __name__ == "__main__":
    main()
