"""Face detection — three backends.

Backend            Config value       Notes
-----------------  -----------------  -------------------------------------------
Short-range        "short_range"      MediaPipe Tasks API, optimised for ≤ ~2 m
Full-range         "full_range"       MediaPipe Solutions API (requires mp ≤ 0.9)
                                      Tasks API does NOT support the full-range
                                      anchor layout (2304 vs 896 anchors)
OpenCV DNN         "opencv"           Res10 SSD MobileNet, works at any distance,
                                      no extra dependencies beyond OpenCV

All backends produce identical DetectedFace output.
FaceLandmarker (speaking detection via jawOpen blendshape) runs on top
of any backend unchanged.

Note on face alignment:
  Short-range and full-range backends provide eye keypoints → affine alignment.
  OpenCV DNN does not provide keypoints → padded-crop fallback (no alignment).
"""

import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from config import (
    DATA_DIR,
    DETECTION_CONFIDENCE,
    FACE_CROP_SIZE,
    FACE_DETECTOR_MODEL,
    FACE_MIN_SIZE,
    SPEAKING_BACKEND,
    SPEAKING_JAW_THRESHOLD,
)
from models import BoundingBox, DetectedFace

try:
    from mediapipe.python.solutions import face_detection as mp_face_detection
except ImportError:
    mp_face_detection = None

# --------------------------------------------------------------------------
# Model files
# --------------------------------------------------------------------------

# --- MediaPipe short-range (Tasks API) ---
_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_DETECTOR_PATH = DATA_DIR / "blaze_face_short_range.tflite"

# --- MediaPipe FaceLandmarker (speaking detection, shared by all backends) ---
_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_LANDMARKER_PATH = DATA_DIR / "face_landmarker.task"

# --- OpenCV DNN — Res10 SSD (auto-downloaded on first run) ---
_OPENCV_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_OPENCV_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)
_OPENCV_PROTO_PATH = DATA_DIR / "opencv_face_detector.prototxt"
_OPENCV_MODEL_PATH = DATA_DIR / "res10_300x300_ssd.caffemodel"

# Canonical eye positions in 112x112 for face alignment (ArcFace standard).
_CANONICAL_EYES = np.float32(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
    ]
)


def _ensure_model(url: str, path):
    if path.exists():
        return
    print(f"Downloading {path.name} …")
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, path)
    print(f"  {path.name} ready.")


# --------------------------------------------------------------------------
# Image-level helpers (shared by all backends)
# --------------------------------------------------------------------------


def _align_face(rgb, eye_right_px, eye_left_px):
    """Similarity-transform so eye centres land on ArcFace canonical positions."""
    src = np.float32([list(eye_right_px), list(eye_left_px)])
    M, _ = cv2.estimateAffinePartial2D(src, _CANONICAL_EYES)
    if M is None:
        return None
    return cv2.warpAffine(
        rgb, M, (FACE_CROP_SIZE, FACE_CROP_SIZE), flags=cv2.INTER_LINEAR
    )


def _blur_score(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _padded_crop(rgb, x1, y1, x2, y2) -> np.ndarray:
    """Fallback 112x112 crop when eye keypoints are unavailable."""
    h, w = rgb.shape[:2]
    pad_w = int((x2 - x1) * 0.2)
    pad_h = int((y2 - y1) * 0.2)
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w, x2 + pad_w)
    cy2 = min(h, y2 + pad_h)
    return cv2.resize(rgb[cy1:cy2, cx1:cx2], (FACE_CROP_SIZE, FACE_CROP_SIZE))


def _build_face(
    rgb,
    x1,
    y1,
    x2,
    y2,
    confidence,
    frame_count,
    timestamp,
    eye_right_px=None,
    eye_left_px=None,
):
    """Validate bbox, align or crop, return DetectedFace or None."""
    h, w = rgb.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < FACE_MIN_SIZE or (y2 - y1) < FACE_MIN_SIZE:
        return None

    crop = None
    if eye_right_px is not None and eye_left_px is not None:
        crop = _align_face(rgb, eye_right_px, eye_left_px)
    if crop is None:
        crop = _padded_crop(rgb, x1, y1, x2, y2)

    return DetectedFace(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence),
        crop=crop,
        frame_index=frame_count,
        timestamp=timestamp,
        blur_score=_blur_score(crop),
    )


# --------------------------------------------------------------------------
# FaceDetector
# --------------------------------------------------------------------------


class FaceDetector:
    """Detects faces and optionally estimates speaking state (FaceLandmarker).

    Args:
        model: "short_range", "full_range", or "opencv".
               Defaults to FACE_DETECTOR_MODEL from config.py.
    """

    def __init__(
        self,
        model: str = FACE_DETECTOR_MODEL,
        min_confidence: float = DETECTION_CONFIDENCE,
    ):
        self._model = model
        self._min_confidence = min_confidence

        _ensure_model(_LANDMARKER_URL, _LANDMARKER_PATH)

        if model == "opencv":
            _ensure_model(_OPENCV_PROTO_URL, _OPENCV_PROTO_PATH)
            _ensure_model(_OPENCV_MODEL_URL, _OPENCV_MODEL_PATH)
            self._detector = cv2.dnn.readNetFromCaffe(
                str(_OPENCV_PROTO_PATH), str(_OPENCV_MODEL_PATH)
            )
            print("[FaceDetector] using OpenCV DNN Res10-SSD (multi-range)")

        elif model == "full_range":
            try:
                if mp_face_detection is None:
                    raise ImportError(
                        "mediapipe.python.solutions.face_detection unavailable"
                    )
                self._detector = mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=min_confidence,
                )
                print("[FaceDetector] using full-range BlazeFace (Solutions API)")
            except (ImportError, AttributeError) as e:
                raise RuntimeError(
                    "full_range requires mediapipe ≤ 0.9 with the Solutions API. "
                    "Your mediapipe version does not have it. "
                    "Use FACE_DETECTOR_MODEL = 'opencv' instead."
                ) from e

        else:  # short_range
            _ensure_model(_DETECTOR_URL, _DETECTOR_PATH)
            detector_opts = mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(_DETECTOR_PATH)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_detection_confidence=min_confidence,
            )
            self._detector = mp.tasks.vision.FaceDetector.create_from_options(
                detector_opts
            )
            print("[FaceDetector] using short-range BlazeFace (Tasks API)")

        # FaceLandmarker — shared by all backends (speaking detection)
        landmarker_opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_LANDMARKER_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=10,
            output_face_blendshapes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            landmarker_opts
        )
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> list[DetectedFace]:
        """Detect faces, align/crop to 112x112, score blur, estimate speaking.

        Args:
            frame: BGR image (H, W, 3) uint8.
        Returns:
            List[DetectedFace], faces smaller than FACE_MIN_SIZE omitted.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._model == "opencv":
            faces = self._detect_opencv(frame, rgb, w, h, timestamp)
        elif self._model == "full_range":
            faces = self._detect_full_range(rgb, w, h, timestamp)
        else:
            faces = self._detect_short_range(rgb, w, h, timestamp)

        if faces and SPEAKING_BACKEND == "mediapipe":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._update_speaking(mp_image, faces, w, h)

        self._frame_count += 1
        return faces

    # ------------------------------------------------------------------
    # Detection backends
    # ------------------------------------------------------------------

    def _detect_opencv(self, frame_bgr, rgb, w, h, timestamp) -> list[DetectedFace]:
        """OpenCV DNN Res10-SSD — works at any range, no eye keypoints."""
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104, 177, 123))
        self._detector.setInput(blob)
        dets = self._detector.forward()  # (1, 1, N, 7)
        faces = []
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < self._min_confidence:
                continue
            face = _build_face(
                rgb,
                x1=dets[0, 0, i, 3] * w,
                y1=dets[0, 0, i, 4] * h,
                x2=dets[0, 0, i, 5] * w,
                y2=dets[0, 0, i, 6] * h,
                confidence=conf,
                frame_count=self._frame_count,
                timestamp=timestamp,
                # No eye keypoints from this detector → _padded_crop fallback
            )
            if face is not None:
                faces.append(face)
        return faces

    def _detect_short_range(self, rgb, w, h, timestamp) -> list[DetectedFace]:
        """MediaPipe Tasks API — BlazeFace short-range (≤ ~2 m)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det_result = self._detector.detect(mp_image)
        faces = []
        for detection in det_result.detections:
            box = detection.bounding_box
            kp = detection.keypoints
            face = _build_face(
                rgb,
                x1=box.origin_x,
                y1=box.origin_y,
                x2=box.origin_x + box.width,
                y2=box.origin_y + box.height,
                confidence=detection.categories[0].score,
                frame_count=self._frame_count,
                timestamp=timestamp,
                eye_right_px=(kp[0].x * w, kp[0].y * h),
                eye_left_px=(kp[1].x * w, kp[1].y * h),
            )
            if face is not None:
                faces.append(face)
        return faces

    def _detect_full_range(self, rgb, w, h, timestamp) -> list[DetectedFace]:
        """MediaPipe Solutions API — BlazeFace full-range (≤ ~5 m)."""
        results = self._detector.process(rgb)
        faces = []
        if not results.detections:
            return faces
        for detection in results.detections:
            bb = detection.location_data.relative_bounding_box
            kps = detection.location_data.relative_keypoints
            face = _build_face(
                rgb,
                x1=bb.xmin * w,
                y1=bb.ymin * h,
                x2=(bb.xmin + bb.width) * w,
                y2=(bb.ymin + bb.height) * h,
                confidence=detection.score[0],
                frame_count=self._frame_count,
                timestamp=timestamp,
                eye_right_px=(kps[0].x * w, kps[0].y * h),
                eye_left_px=(kps[1].x * w, kps[1].y * h),
            )
            if face is not None:
                faces.append(face)
        return faces

    # ------------------------------------------------------------------
    # Speaking detection — FaceLandmarker (shared by all backends)
    # ------------------------------------------------------------------

    def _update_speaking(self, mp_image, faces, w, h):
        lm_result = self._landmarker.detect(mp_image)
        if not lm_result.face_landmarks or not lm_result.face_blendshapes:
            return
        for landmarks, blendshapes in zip(
            lm_result.face_landmarks, lm_result.face_blendshapes, strict=False
        ):
            cx = float(np.mean([lm.x for lm in landmarks])) * w
            cy = float(np.mean([lm.y for lm in landmarks])) * h
            jaw_open = next(
                (bs.score for bs in blendshapes if bs.category_name == "jawOpen"), 0.0
            )
            best_face, best_dist = None, float("inf")
            for face in faces:
                fc_x, fc_y = face.bbox.center
                dist = ((cx - fc_x) ** 2 + (cy - fc_y) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_face = dist, face
            if best_face is not None:
                max_dist = max(best_face.bbox.width, best_face.bbox.height)
                if best_dist < max_dist:
                    best_face.is_speaking = jaw_open > SPEAKING_JAW_THRESHOLD

    # ------------------------------------------------------------------

    def close(self):
        if self._model != "opencv":
            self._detector.close()
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
