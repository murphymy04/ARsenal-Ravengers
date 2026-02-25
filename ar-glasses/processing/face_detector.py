"""Face detection using OpenCV Haar Cascade.

Detects faces in BGR frames and returns cropped 112x112 RGB images
suitable for embedding extraction.
"""

import cv2
import numpy as np
from typing import List

from models import BoundingBox, DetectedFace
from config import FACE_CROP_SIZE


class FaceDetector:
    """Detects faces using OpenCV Haar Cascade and produces 112x112 crops."""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        self._frame_count = 0

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> List[DetectedFace]:
        """Detect faces in a BGR frame.

        Args:
            frame: BGR image (H, W, 3) uint8.
            timestamp: frame timestamp in seconds.

        Returns:
            List of DetectedFace with 112x112 RGB crops.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        faces: List[DetectedFace] = []
        if len(detections) == 0:
            self._frame_count += 1
            return faces

        for (x, y, fw, fh) in detections:
            x1, y1, x2, y2 = x, y, x + fw, y + fh

            # Add 10% padding around the face
            pad = int(min(fw, fh) * 0.1)
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)

            crop_bgr = frame[cy1:cy2, cx1:cx2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_112 = cv2.resize(crop_rgb, (FACE_CROP_SIZE, FACE_CROP_SIZE))

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=1.0)
            faces.append(DetectedFace(
                bbox=bbox,
                crop=crop_112,
                frame_index=self._frame_count,
                timestamp=timestamp,
            ))

        self._frame_count += 1
        return faces

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
