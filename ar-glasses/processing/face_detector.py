"""Face detection using MediaPipe Face Detector Tasks API.

Uses BlazeFace short-range model for fast face detection and alignment,
plus FaceLandmarker for speaking detection via the jawOpen blendshape.

Improvements:
  - Face alignment: eye keypoints warped to canonical 112×112 positions (#1).
  - Size filter: detections smaller than FACE_MIN_SIZE are ignored (#2).
  - Blur score: Laplacian variance so callers can skip blurry frames (#2/#8).
  - Speaking detection: FaceLandmarker jawOpen blendshape > threshold.
"""

import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from typing import List

from models import BoundingBox, DetectedFace
from config import (
    DETECTION_CONFIDENCE, FACE_CROP_SIZE, FACE_MIN_SIZE, SPEAKING_JAW_THRESHOLD,
    DATA_DIR, SPEAKING_BACKEND,
)

# --------------------------------------------------------------------------
# Model files
# --------------------------------------------------------------------------

_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
_DETECTOR_PATH = DATA_DIR / "blaze_face_short_range.tflite"

_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_LANDMARKER_PATH = DATA_DIR / "face_landmarker.task"

# Canonical eye positions in 112×112 for face alignment (ArcFace standard).
# Index 0 = person's right eye (image left), index 1 = person's left eye (image right).
_CANONICAL_EYES = np.float32([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
])


def _ensure_model(url: str, path):
    if path.exists():
        return
    print(f"Downloading {path.name} …")
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, path)
    print(f"  {path.name} ready.")


# --------------------------------------------------------------------------
# Image-level helpers
# --------------------------------------------------------------------------

def _align_face(
    rgb: np.ndarray,
    eye_right_px: tuple[float, float],
    eye_left_px: tuple[float, float],
) -> np.ndarray | None:
    """Similarity-transform the frame so eye centres land on canonical positions."""
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
    """Fallback 112×112 crop when alignment fails."""
    h, w = rgb.shape[:2]
    pad_w = int((x2 - x1) * 0.2)
    pad_h = int((y2 - y1) * 0.2)
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w, x2 + pad_w)
    cy2 = min(h, y2 + pad_h)
    return cv2.resize(rgb[cy1:cy2, cx1:cx2], (FACE_CROP_SIZE, FACE_CROP_SIZE))


# --------------------------------------------------------------------------
# FaceDetector
# --------------------------------------------------------------------------

class FaceDetector:
    """Detects faces (BlazeFace) and estimates speaking state (FaceLandmarker)."""

    def __init__(self, min_confidence: float = DETECTION_CONFIDENCE):
        _ensure_model(_DETECTOR_URL,   _DETECTOR_PATH)
        _ensure_model(_LANDMARKER_URL, _LANDMARKER_PATH)

        # BlazeFace — VIDEO mode uses temporal smoothing across frames
        detector_opts = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_DETECTOR_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(detector_opts)

        # FaceLandmarker — VIDEO mode for temporal consistency
        landmarker_opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_LANDMARKER_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=10,
            output_face_blendshapes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(landmarker_opts)

        self._frame_count = 0

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> List[DetectedFace]:
        """Detect faces, align crops, score blur, and estimate speaking state.

        Args:
            frame: BGR image (H, W, 3) uint8.
            timestamp: frame timestamp in seconds.

        Returns:
            List of DetectedFace. Faces smaller than FACE_MIN_SIZE are omitted.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(timestamp * 1000)

        # --- BlazeFace detection ---
        det_result = self._detector.detect_for_video(mp_image, timestamp_ms)
        faces: List[DetectedFace] = []

        for detection in det_result.detections:
            box = detection.bounding_box
            x1 = max(0, box.origin_x)
            y1 = max(0, box.origin_y)
            x2 = min(w, box.origin_x + box.width)
            y2 = min(h, box.origin_y + box.height)

            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < FACE_MIN_SIZE or (y2 - y1) < FACE_MIN_SIZE:
                continue

            confidence = detection.categories[0].score
            kp = detection.keypoints
            eye_right_px = (kp[0].x * w, kp[0].y * h)
            eye_left_px  = (kp[1].x * w, kp[1].y * h)

            crop_112 = _align_face(rgb, eye_right_px, eye_left_px)
            if crop_112 is None:
                crop_112 = _padded_crop(rgb, x1, y1, x2, y2)

            faces.append(DetectedFace(
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence),
                crop=crop_112,
                frame_index=self._frame_count,
                timestamp=timestamp,
                blur_score=_blur_score(crop_112),
            ))

        # --- Speaking detection (FaceLandmarker) ---
        # Skipped when Light-ASD is active; Light-ASD sets is_speaking externally.
        # How does this get skipped? the speaking backend is a constant.
        if faces and SPEAKING_BACKEND == "mediapipe":
            self._update_speaking(mp_image, timestamp_ms, faces, w, h)

        self._frame_count += 1
        return faces

    def _update_speaking(
        self,
        mp_image: mp.Image,
        timestamp_ms: int,
        faces: List[DetectedFace],
        w: int,
        h: int,
    ):
        """Run FaceLandmarker and set is_speaking on the best-matching face."""
        lm_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not lm_result.face_landmarks or not lm_result.face_blendshapes:
            return

        for landmarks, blendshapes in zip(
            lm_result.face_landmarks, lm_result.face_blendshapes
        ):
            # Landmark centroid in pixels — used to match to a BlazeFace bbox
            cx = float(np.mean([lm.x for lm in landmarks])) * w
            cy = float(np.mean([lm.y for lm in landmarks])) * h

            # jawOpen blendshape: 0 = closed, 1 = fully open
            jaw_open = next(
                (bs.score for bs in blendshapes if bs.category_name == "jawOpen"),
                0.0,
            )

            # Match to the nearest BlazeFace detection
            best_face, best_dist = None, float("inf")
            for face in faces:
                fc_x, fc_y = face.bbox.center
                dist = ((cx - fc_x) ** 2 + (cy - fc_y) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_face = dist, face

            # Only assign if centroid is within the face bbox (sanity check)
            if best_face is not None:
                max_dist = max(best_face.bbox.width, best_face.bbox.height)
                if best_dist < max_dist:
                    best_face.is_speaking = jaw_open > SPEAKING_JAW_THRESHOLD

    def close(self):
        self._detector.close()
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
