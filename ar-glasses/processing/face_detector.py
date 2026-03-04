"""Face detection using MediaPipe Face Detector Tasks API.

Uses BlazeFace short-range model via the mediapipe.tasks API (compatible
with mediapipe 0.10.20+). Model is auto-downloaded on first run (~800KB).

Improvements over naive crop-and-resize:
  - Face alignment: eye keypoints are warped to canonical 112×112 positions
    so the embedder always sees a geometrically consistent face (#1).
  - Size filter: detections smaller than FACE_MIN_SIZE are ignored (#2).
  - Blur score: each DetectedFace carries a Laplacian variance score so
    callers can decide whether to use the crop for embedding (#2/#8).
"""

import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from typing import List

from models import BoundingBox, DetectedFace
from config import FACE_CROP_SIZE, FACE_MIN_SIZE, DATA_DIR

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
MODEL_PATH = DATA_DIR / "blaze_face_short_range.tflite"

# Canonical eye positions in 112×112 (ArcFace / EdgeFace standard).
# Index 0 = person's right eye (appears on the LEFT of the image, smaller x).
# Index 1 = person's left  eye (appears on the RIGHT of the image, larger x).
_CANONICAL_EYES = np.float32([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
])


def _ensure_model():
    """Download the BlazeFace model if not already present."""
    if MODEL_PATH.exists():
        return
    print(f"Downloading BlazeFace model to {MODEL_PATH} ...")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def _align_face(
    rgb: np.ndarray,
    eye_right_px: tuple[float, float],
    eye_left_px: tuple[float, float],
) -> np.ndarray | None:
    """Warp the full frame to produce a canonical 112×112 face crop.

    Uses a similarity transform (rotation + scale + translation, no shear)
    so that the two detected eye centres land on the ArcFace canonical
    positions.  Returns None if the transform cannot be estimated.
    """
    src = np.float32([list(eye_right_px), list(eye_left_px)])
    M, _ = cv2.estimateAffinePartial2D(src, _CANONICAL_EYES)
    if M is None:
        return None
    return cv2.warpAffine(rgb, M, (FACE_CROP_SIZE, FACE_CROP_SIZE), flags=cv2.INTER_LINEAR)


def _blur_score(crop: np.ndarray) -> float:
    """Laplacian variance of a crop — higher means sharper."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _padded_crop(rgb: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    """Fallback: padded bounding-box crop resized to 112×112."""
    h, w = rgb.shape[:2]
    pad_w = int((x2 - x1) * 0.2)
    pad_h = int((y2 - y1) * 0.2)
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w, x2 + pad_w)
    cy2 = min(h, y2 + pad_h)
    return cv2.resize(rgb[cy1:cy2, cx1:cx2], (FACE_CROP_SIZE, FACE_CROP_SIZE))


class FaceDetector:
    """Detects faces using MediaPipe BlazeFace and produces 112×112 aligned crops."""

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
            List of DetectedFace with aligned 112×112 RGB crops and blur scores.
            Faces smaller than FACE_MIN_SIZE on either axis are skipped.
        """
        h, w = frame.shape[:2]

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

            # Skip faces that are too small to produce a useful embedding (#2)
            if (x2 - x1) < FACE_MIN_SIZE or (y2 - y1) < FACE_MIN_SIZE:
                continue

            confidence = detection.categories[0].score

            # Align using eye keypoints — kp[0] = person's right eye, kp[1] = left (#1)
            kp = detection.keypoints
            eye_right_px = (kp[0].x * w, kp[0].y * h)
            eye_left_px  = (kp[1].x * w, kp[1].y * h)
            crop_112 = _align_face(rgb, eye_right_px, eye_left_px)

            if crop_112 is None:
                # Fallback: padded bbox crop
                crop_112 = _padded_crop(rgb, x1, y1, x2, y2)

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)
            faces.append(DetectedFace(
                bbox=bbox,
                crop=crop_112,
                frame_index=self._frame_count,
                timestamp=timestamp,
                blur_score=_blur_score(crop_112),
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
