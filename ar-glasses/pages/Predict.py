"""Prediction page — live camera feed with continuous identity prediction."""

import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import streamlit as st

from config import MATCH_THRESHOLD
from input.camera import Camera
from processing.face_detector import FaceDetector
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from storage.database import Database


# ---------------------------------------------------------------------------
# Shared pipeline state (mirrors demo.py pattern)
# ---------------------------------------------------------------------------


class _PredictState:
    def __init__(self):
        self._lock = threading.Lock()
        self.annotated_frame: Optional[np.ndarray] = None
        self.face_predictions: list[dict] = []
        self.fps: float = 0.0
        self.running = True
        self.stop_event = threading.Event()
        self.error: Optional[str] = None
        self.status: str = "Starting…"

    def update(self, annotated: np.ndarray, predictions: list[dict], fps: float):
        with self._lock:
            self.annotated_frame = annotated
            self.face_predictions = predictions
            self.fps = fps

    def snapshot(self):
        with self._lock:
            ann = (
                self.annotated_frame.copy()
                if self.annotated_frame is not None
                else None
            )
            return ann, list(self.face_predictions), self.fps


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------


def _draw_predictions(
    frame: np.ndarray, faces, candidates_per_face: list
) -> np.ndarray:
    out = frame.copy()
    for face, candidates in zip(faces, candidates_per_face):
        b = face.bbox
        if not candidates:
            continue

        top_person, top_score = candidates[0]
        second_score = candidates[1][1] if len(candidates) > 1 else 0.0
        margin = top_score - second_score
        is_known = top_score >= MATCH_THRESHOLD

        color = (0, 220, 60) if is_known else (30, 80, 220)

        overlay = out.copy()
        cv2.rectangle(overlay, (b.x1, b.y1), (b.x2, b.y2), color, -1)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), color, 2)

        name = top_person.name if is_known else "Unknown"
        label = f"{name}  {top_score:.2f} (±{margin:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (b.x1, b.y1 - th - 12), (b.x1 + tw + 8, b.y1), color, -1)
        cv2.putText(
            out,
            label,
            (b.x1 + 4, b.y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    return out


# ---------------------------------------------------------------------------
# Background pipeline thread
# ---------------------------------------------------------------------------


def _pipeline_thread(state: _PredictState):
    try:
        state.status = "Loading models…"
        db = Database()
        detector = FaceDetector()
        embedder = FaceEmbedder()
        matcher = FaceMatcher()

        people = db.get_all_people()
        matcher.update_gallery(people)
        db.close()

        state.status = "Starting camera…"
        camera = Camera()
    except Exception as e:
        state.error = str(e)
        state.running = False
        return

    frame_count = 0
    start_time = time.time()

    try:
        state.status = "Running"
        for frame in camera.frames():
            if state.stop_event.is_set():
                break

            faces = detector.detect(frame, timestamp=time.time())

            candidates_per_face = []
            for face in faces:
                embedding = embedder.embed(face.crop)
                candidates = matcher.rank_candidates(embedding)
                candidates_per_face.append(candidates)

            annotated = _draw_predictions(frame, faces, candidates_per_face)

            predictions = [
                {
                    "name": c[0][0].name
                    if c and c[0][1] >= MATCH_THRESHOLD
                    else "Unknown",
                    "score": round(c[0][1], 3) if c else 0.0,
                    "margin": round(c[0][1] - c[1][1], 3) if len(c) > 1 else 0.0,
                    "candidates": [(p.name, round(s, 3)) for p, s in c[:3]],
                }
                for c in candidates_per_face
            ]

            fps = frame_count / max(time.time() - start_time, 1e-6)
            state.update(annotated, predictions, fps)
            frame_count += 1

    except Exception as e:
        state.error = str(e)
    finally:
        camera.close()
        detector.close()
        state.running = False


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _ensure_pipeline_running():
    def _state_valid(s) -> bool:
        return isinstance(s, _PredictState) and hasattr(s, "face_predictions")

    if "predict_state" not in st.session_state or not _state_valid(
        st.session_state["predict_state"]
    ):
        predict_state = _PredictState()
        thread = threading.Thread(
            target=_pipeline_thread, args=(predict_state,), daemon=True
        )
        thread.start()
        st.session_state["predict_state"] = predict_state
        st.session_state["predict_thread"] = thread


# ---------------------------------------------------------------------------
# Resize helper (mirrors demo.py)
# ---------------------------------------------------------------------------


def _resize(img: np.ndarray, width: int = 640, max_height: int = 840) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, max_height / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ARsenal — Predict", layout="wide", page_icon="🔍")
st.title("🔍 Live Prediction")

fps_ph = st.sidebar.empty()
st.sidebar.markdown("**Green box** = confident match  \n**Blue box** = unknown")

feed_ph = st.empty()

st.divider()
st.subheader("Detected faces")
faces_ph = st.empty()

_ensure_pipeline_running()
state: _PredictState = st.session_state["predict_state"]

while True:
    if state.error:
        st.error(f"Pipeline error: {state.error}")
        break
    if not state.running:
        st.info("Pipeline stopped.")
        break

    annotated, predictions, fps = state.snapshot()

    if annotated is None:
        feed_ph.caption(f"⏳ {state.status}")
        time.sleep(0.1)
        continue

    feed_ph.image(
        cv2.cvtColor(_resize(annotated), cv2.COLOR_BGR2RGB),
        use_container_width=False,
        width=640,
    )

    with faces_ph.container():
        if predictions:
            cols = st.columns(max(len(predictions), 1))
            for i, pred in enumerate(predictions):
                with cols[i]:
                    if pred["name"] != "Unknown":
                        st.success(f"**{pred['name']}**")
                    else:
                        st.warning("**Unknown**")
                    st.caption(
                        f"score: `{pred['score']:.3f}`  margin: `{pred['margin']:.3f}`"
                    )
                    for rank, (name, score) in enumerate(pred["candidates"], 1):
                        above = score >= MATCH_THRESHOLD
                        st.write(
                            f"{'🟢' if above else '🔴'} #{rank} {name} `{score:.3f}`"
                        )
                        st.progress(float(np.clip(score, 0.0, 1.0)))
        else:
            st.caption("No faces in frame")

    fps_ph.metric("FPS", f"{fps:.1f}")
    time.sleep(0.05)
