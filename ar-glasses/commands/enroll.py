"""CLI command for manually enrolling a named person via webcam."""

import cv2

from input.camera import Camera
from processing.face_detector import FaceDetector
from processing.face_embedder import FaceEmbedder
from storage.database import Database
from storage.enrollment import Enrollment


def enroll_mode(db: Database, detector: FaceDetector, embedder: FaceEmbedder):
    """Capture webcam frames and enroll a named person into the database."""
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
        for face in detector.detect(frame):
            b = face.bbox
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Enrolling: {name} | Captured: {len(captured_images)} | 'c'=capture 'q'=done",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        cv2.imshow("Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            if detector.detect(frame):
                captured_images.append(frame.copy())
                print(f"  Captured image {len(captured_images)}")
        elif key == ord("q"):
            break

    camera.close()
    cv2.destroyAllWindows()

    if not captured_images:
        print("No images captured.")
        return

    person_id = enrollment.enroll_from_images(name, captured_images)
    if person_id:
        print(f"Enrolled '{name}' with ID {person_id} ({len(captured_images)} images)")
    else:
        print("Failed to detect a face in any captured image.")
