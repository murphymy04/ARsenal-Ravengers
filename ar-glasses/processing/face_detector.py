"""Face detection using MediaPipe Face Detector Tasks API.

Uses BlazeFace short-range model via the mediapipe.tasks API (compatible
with mediapipe 0.10.20+). Model is auto-downloaded on first run (~800KB).
"""

import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from typing import List

from models import BoundingBox, DetectedFace
from config import FACE_CROP_SIZE, DATA_DIR

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
MODEL_PATH = DATA_DIR / "blaze_face_short_range.tflite"


def _ensure_model():
    """Download the BlazeFace model if not already present."""
    if MODEL_PATH.exists():
        return
    print(f"Downloading BlazeFace model to {MODEL_PATH} ...")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


class FaceDetector:
    """Detects faces using MediaPipe BlazeFace and produces 112x112 crops."""

    def __init__(self, min_confidence: float = 0.5):
        _ensure_model()

        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)
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

        # MediaPipe Tasks expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        faces: List[DetectedFace] = []
        for detection in result.detections:
            box = detection.bounding_box
            x1 = max(0, box.origin_x)
            y1 = max(0, box.origin_y)
            x2 = min(w, box.origin_x + box.width)
            y2 = min(h, box.origin_y + box.height)

            if x2 <= x1 or y2 <= y1:
                continue

            confidence = detection.categories[0].score

            # Add 20% padding for better embedding quality
            pad_w = int((x2 - x1) * 0.2)
            pad_h = int((y2 - y1) * 0.2)
            cx1 = max(0, x1 - pad_w)
            cy1 = max(0, y1 - pad_h)
            cx2 = min(w, x2 + pad_w)
            cy2 = min(h, y2 + pad_h)

            crop_rgb = rgb[cy1:cy2, cx1:cx2]
            crop_112 = cv2.resize(crop_rgb, (FACE_CROP_SIZE, FACE_CROP_SIZE))

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)
            faces.append(DetectedFace(
                bbox=bbox,
                crop=crop_112,
                frame_index=self._frame_count,
                timestamp=timestamp,
            ))

        self._frame_count += 1
        return faces

    def close(self):
        """Release MediaPipe resources."""
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
