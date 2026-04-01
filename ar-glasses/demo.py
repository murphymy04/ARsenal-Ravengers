"""AR Glasses — Streamlit demo.

Run with:
    streamlit run demo.py
    streamlit run demo.py -- --camera android
    streamlit run demo.py -- --camera android --mic-device 2
    streamlit run demo.py -- --camera ipwebcam
    streamlit run demo.py -- --camera ipwebcam --ipwebcam-url http://localhost:9090/video

IP Webcam setup (Play Store app):
    1. Install 'IP Webcam' on the glasses, tap 'Start server'.
    2. adb forward tcp:8080 tcp:8080
    3. streamlit run demo.py -- --camera ipwebcam

To list available microphone devices:
    python -c "import sounddevice as sd; print(sd.query_devices())"
"""

import io
import sys
import time
import wave
import threading
import collections
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import streamlit as st

from config import (
    DB_PATH,
    PENDING_EXPIRY_FRAMES,
    SPEAKING_BACKEND,
    LIVE_BUFFER_SECONDS,
    SAMPLE_RATE,
)
from input.camera import Camera
from input.microphone import Microphone
from processing.face_detector import FaceDetector
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from processing.face_tracker import FaceTracker
from storage.database import Database
from storage.speaking_log import SpeakingLog
from main import _maybe_store_embedding, _update_pending

try:
    from pipeline.transcription import TranscriptionPipeline

    _TRANSCRIPTION_AVAILABLE = True
except Exception as _e:
    _TRANSCRIPTION_AVAILABLE = False
    _TRANSCRIPTION_ERROR = str(_e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Whisper hallucinates these phrases on silence — filter them out
_HALLUCINATIONS = {
    "thank you.",
    "thank you",
    "thanks.",
    "thanks",
    "you.",
    ".",
    "..",
    "...",
    "bye.",
    "bye",
    "you",
    "okay.",
}


def _pcm_to_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


class PipelineState:
    def __init__(self):
        self._lock = threading.Lock()
        self.raw_frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self.current_faces: list[dict] = []
        self.detection_log: collections.deque = collections.deque(maxlen=50)
        self.transcript: collections.deque = collections.deque(maxlen=200)
        self.fps: float = 0.0
        self.running: bool = True
        self.stop_event = threading.Event()
        self.error: Optional[str] = None
        self.mic_status: str = "not started"
        self.status: str = "Starting…"

    def update(self, raw, annotated, face_dicts, fps):
        with self._lock:
            self.raw_frame = raw
            self.annotated_frame = annotated
            self.current_faces = face_dicts
            self.fps = fps

    def append_log(self, entry: dict):
        with self._lock:
            self.detection_log.appendleft(entry)

    def append_transcript(self, segments: list[dict]):
        with self._lock:
            self.transcript.extend(segments)

    def snapshot(self):
        with self._lock:
            return (
                self.raw_frame.copy() if self.raw_frame is not None else None,
                self.annotated_frame.copy()
                if self.annotated_frame is not None
                else None,
                list(self.current_faces),
                list(self.detection_log),
                list(self.transcript),
                self.fps,
                self.mic_status,
            )


# ---------------------------------------------------------------------------
# Background pipeline thread
# ---------------------------------------------------------------------------


def _recognition_thread(
    state: PipelineState,
    camera_source,
    mic_device: Optional[int],
    ipwebcam_url: str = "http://localhost:8080/video",
):
    try:
        state.status = "Loading face detector…"
        db = Database()
        detector = FaceDetector()
        state.status = "Loading face embedder…"
        embedder = FaceEmbedder()
        matcher = FaceMatcher()
        tracker = FaceTracker()
        state.status = "Loading speaking detector…"
        speaking_det = _init_speaking_detector()
        speaking_log = SpeakingLog()
        log_path = DB_PATH.parent / f"speaking_log_{int(time.time())}.json"

        people = db.get_all_people()
        matcher.update_gallery(people)

        state.status = "Starting camera…"
        if camera_source == "android":
            from input.android_camera import AndroidCamera

            camera = AndroidCamera()
        elif camera_source == "ipwebcam":
            from input.android_camera import IPWebcamCamera

            camera = IPWebcamCamera(url=ipwebcam_url)
        else:
            camera = Camera(source=camera_source)

    except Exception as e:
        state.error = str(e)
        state.running = False
        return

    # Microphone setup
    mic: Optional[Microphone] = None
    transcriber = None
    if _TRANSCRIPTION_AVAILABLE:
        try:
            mic = Microphone(device=mic_device)
            mic.open()
            label = f"device {mic_device}" if mic_device is not None else "default"
            state.mic_status = f"active ({label})"
            transcriber = TranscriptionPipeline()
        except Exception as e:
            state.mic_status = f"error: {e}"
            mic = None

    pending: list[dict] = []
    last_embedding_update: dict[int, int] = {}
    seen_tracks: dict[int, str] = {}
    speaking_tracks: set[int] = set()
    frame_count = 0
    start_time = time.time()

    # Rolling transcription window
    window_log = SpeakingLog()
    last_flush = time.time()
    stream_offset = time.time()  # used to label transcript timestamps

    try:
        state.status = "Running"
        frames_received = 0
        for frame in camera.frames():
            if state.stop_event.is_set():
                break
            frames_received += 1

            timestamp = time.time()
            faces = detector.detect(frame, timestamp=timestamp)

            pending = [
                p
                for p in pending
                if frame_count - p["last_frame"] < PENDING_EXPIRY_FRAMES
            ]

            raw_matches, gallery_dirty = [], False
            for face in faces:
                embedding = embedder.embed(face.crop)
                match = matcher.match(embedding)
                if match.is_known:
                    gallery_dirty |= _maybe_store_embedding(
                        db,
                        match.person_id,
                        embedding,
                        face,
                        last_embedding_update,
                        frame_count,
                    )
                else:
                    match, promoted = _update_pending(
                        db,
                        embedding,
                        face,
                        pending,
                        last_embedding_update,
                        frame_count,
                    )
                    gallery_dirty |= promoted
                raw_matches.append(match)

            if gallery_dirty:
                matcher.update_gallery(db.get_all_people())

            matches, track_ids, _ = tracker.update(faces, raw_matches, frame_count)

            if speaking_det is not None:
                for face, tid in zip(faces, track_ids):
                    speaking_det.add_crop(tid, face.crop)
                speaking_det.run_inference(frame_count, active_track_ids=set(track_ids))

            ts = time.time()
            rel_ts = ts - stream_offset
            for face, match, tid in zip(faces, matches, track_ids):
                if speaking_det is not None:
                    face.is_speaking = speaking_det.get_speaking(tid)
                speaking_log.update(
                    track_id=tid,
                    person_id=match.person_id if match.is_known else None,
                    name=match.name,
                    is_speaking=face.is_speaking,
                    timestamp=rel_ts,
                )
                window_log.update(
                    track_id=tid,
                    person_id=match.person_id if match.is_known else None,
                    name=match.name,
                    is_speaking=face.is_speaking,
                    timestamp=rel_ts,
                )
                if tid not in seen_tracks:
                    state.append_log(
                        {
                            "time": time.strftime("%H:%M:%S"),
                            "event": "spotted",
                            "name": match.name,
                            "is_known": match.is_known,
                        }
                    )
                    seen_tracks[tid] = match.name
                if face.is_speaking and tid not in speaking_tracks:
                    state.append_log(
                        {
                            "time": time.strftime("%H:%M:%S"),
                            "event": "speaking",
                            "name": match.name,
                            "is_known": match.is_known,
                        }
                    )
                    speaking_tracks.add(tid)
                elif not face.is_speaking:
                    speaking_tracks.discard(tid)

            active = set(track_ids)
            for tid in list(seen_tracks):
                if tid not in active:
                    del seen_tracks[tid]

            # Flush transcription window
            if mic is not None and transcriber is not None:
                elapsed = time.time() - last_flush
                if elapsed >= LIVE_BUFFER_SECONDS:
                    window_start_offset = rel_ts - elapsed
                    _flush_transcription(
                        state,
                        mic,
                        window_log,
                        transcriber,
                        window_start_offset,
                    )
                    window_log = SpeakingLog()
                    last_flush = time.time()

            annotated = _draw_faces(frame, faces, matches)
            face_dicts = [
                {
                    "name": m.name,
                    "is_known": m.is_known,
                    "is_speaking": f.is_speaking,
                }
                for f, m in zip(faces, matches)
            ]

            fps = frame_count / max(time.time() - start_time, 1e-6)
            state.update(frame, annotated, face_dicts, fps)
            frame_count += 1

    except Exception as e:
        state.error = str(e)
    finally:
        if not state.error and frames_received == 0:
            if camera_source == "ipwebcam":
                state.error = (
                    f"IP Webcam stream produced no frames — check setup:\n"
                    f"  1. IP Webcam app is running on the glasses\n"
                    f"  2. adb forward tcp:8080 tcp:8080\n"
                    f"  3. Stream URL: {ipwebcam_url}"
                )
            else:
                state.error = (
                    "Camera produced no frames — check ADB connection.\n"
                    "Run: adb devices"
                )
        speaking_log.save(log_path)
        if mic is not None:
            mic.close()
        camera.close()
        detector.close()
        if speaking_det:
            speaking_det.close()
        db.close()
        state.running = False


def _flush_transcription(
    state: PipelineState,
    mic: Microphone,
    window_log: SpeakingLog,
    transcriber,
    window_start_offset: float,
):
    audio = mic.get_buffer_and_clear()
    if len(audio) < SAMPLE_RATE:  # less than 1 second of audio
        return

    # Skip near-silent windows — Whisper hallucinates on quiet audio
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 0.01:
        return

    window_log.close()
    try:
        wav_bytes = _pcm_to_wav(audio)
        segments = transcriber.run(wav_bytes)
        seg_dicts = [
            {"start": s.start_time, "end": s.end_time, "text": s.text} for s in segments
        ]
        labeled = window_log.assign_transcript(seg_dicts)

        # Make timestamps relative to stream start
        for seg in labeled:
            seg["start"] = round(seg["start"] + window_start_offset, 1)
            seg["end"] = round(seg["end"] + window_start_offset, 1)

        # Filter hallucinations and consecutive duplicates
        last_text = state.transcript[-1]["text"] if state.transcript else ""
        filtered = []
        for seg in labeled:
            text = seg["text"].strip()
            if text.lower() in _HALLUCINATIONS:
                continue
            if text == last_text:
                continue
            filtered.append(seg)
            last_text = text

        if filtered:
            state.append_transcript(filtered)
            print(f"\n[Transcription] +{len(filtered)} segments")
    except Exception as e:
        print(f"\n[Transcription] Error: {e}")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_faces(frame: np.ndarray, faces, matches) -> np.ndarray:
    out = frame.copy()
    for face, match in zip(faces, matches):
        b = face.bbox
        color = (0, 220, 60) if match.is_known else (30, 80, 220)

        overlay = out.copy()
        cv2.rectangle(overlay, (b.x1, b.y1), (b.x2, b.y2), color, -1)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

        thickness = 4 if face.is_speaking else 2
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), color, thickness)

        if face.is_speaking:
            cx = (b.x1 + b.x2) // 2
            cy = b.y1
            cv2.circle(out, (cx, cy - 18), 14, (0, 255, 255), 2)
            cv2.putText(
                out,
                "SPK",
                (cx - 14, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

        label = match.name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (b.x1, b.y1 - th - 12), (b.x1 + tw + 8, b.y1), color, -1)
        cv2.putText(
            out,
            label,
            (b.x1 + 4, b.y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
    return out


def _init_speaking_detector():
    if SPEAKING_BACKEND != "light_asd":
        return None
    try:
        from processing.speaking_detector import SpeakingDetector

        return SpeakingDetector()
    except Exception as e:
        print(f"Light-ASD init failed ({e})")
        return None


def _resize(img: np.ndarray, width: int = 640, max_height: int = 840) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, max_height / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def _to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ARgusEye", layout="wide", page_icon="🕶️")

# Parse args (Streamlit strips `--` before the script sees sys.argv)
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--camera",
    default="0",
    help="Camera source: 0 (webcam), android (scrcpy), ipwebcam (IP Webcam app)",
)
_parser.add_argument(
    "--ipwebcam-url",
    default="http://localhost:8080/video",
    help="IP Webcam MJPEG stream URL (used when --camera ipwebcam)",
)
_parser.add_argument(
    "--mic-device",
    type=int,
    default=None,
    help="sounddevice index for microphone (glasses USB audio)",
)
_parsed, _ = _parser.parse_known_args(sys.argv[1:])
_camera_source = (
    "android"
    if _parsed.camera == "android"
    else "ipwebcam"
    if _parsed.camera == "ipwebcam"
    else int(_parsed.camera)
)
_ipwebcam_url = _parsed.ipwebcam_url
_mic_device = _parsed.mic_device


# Start pipeline once per browser session.
# Also restart if the stored state is from an older version (missing fields after hot-reload).
def _state_is_valid(s) -> bool:
    return (
        isinstance(s, PipelineState)
        and hasattr(s, "transcript")
        and hasattr(s, "mic_status")
    )


if "pipeline_state" not in st.session_state or not _state_is_valid(
    st.session_state["pipeline_state"]
):
    _state = PipelineState()
    _thread = threading.Thread(
        target=_recognition_thread,
        args=(_state, _camera_source, _mic_device, _ipwebcam_url),
        daemon=True,
    )
    _thread.start()
    st.session_state["pipeline_state"] = _state
    st.session_state["pipeline_thread"] = _thread

state: PipelineState = st.session_state["pipeline_state"]

# Sidebar
fps_ph = st.sidebar.empty()
mic_ph = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Green box** = known person  \n**Blue box** = unknown  \n**Cyan ring** = speaking"
)
if not _TRANSCRIPTION_AVAILABLE:
    st.sidebar.warning(f"Transcription unavailable: {_TRANSCRIPTION_ERROR}")

st.title("🕶️ ARgusEye")

st.subheader("Face Recognition")
rec_ph = st.empty()

st.divider()
st.subheader("Speaking Detection")
speaking_ph = st.empty()

st.divider()
st.subheader("Live Transcript")
transcript_ph = st.empty()

st.divider()
st.subheader("Detection Log")
log_ph = st.empty()

# ---------------------------------------------------------------------------
# Real-time update loop (drives Tab 1 placeholders)
# ---------------------------------------------------------------------------
while True:
    if state.error:
        st.error(f"Pipeline error: {state.error}")
        break
    if not state.running:
        st.info("Pipeline stopped.")
        break

    raw, annotated, faces, log, transcript, fps, mic_status = state.snapshot()

    if raw is None:
        rec_ph.caption(f"⏳ {state.status}")
        time.sleep(0.1)
        continue

    rec_ph.image(_to_rgb(_resize(annotated if annotated is not None else raw)))

    # Speaking cards
    with speaking_ph.container():
        if faces:
            cols = st.columns(max(len(faces), 1))
            for i, face in enumerate(faces):
                with cols[i]:
                    if face["is_speaking"]:
                        st.success(f"🎤 **{face['name']}**\nSPEAKING")
                    else:
                        st.info(f"🔇 **{face['name']}**\nsilent")
        else:
            st.caption("No faces in frame")

    # Transcript
    with transcript_ph.container():
        if transcript:
            shown = list(reversed(transcript[-20:]))
            rows = [
                f"| {_fmt_time(s['start'])} | **{s['speaker']}** | {s['text']} |"
                for s in shown
            ]
            st.markdown(
                "| Time | Speaker | Text |\n|------|---------|------|\n"
                + "\n".join(rows),
                unsafe_allow_html=False,
            )
        else:
            if _TRANSCRIPTION_AVAILABLE:
                st.caption(
                    f"Waiting for speech… (flushes every {LIVE_BUFFER_SECONDS}s)"
                )
            else:
                st.caption("Transcription disabled — GROQ_API_KEY not set in .env")

    # Detection log
    if log:
        icons = {"spotted": "👁️", "speaking": "🎤"}
        rows = [
            f"| {e['time']} | {icons.get(e['event'], '')} {e['event']} | "
            f"{'✅' if e['is_known'] else '❓'} {e['name']} |"
            for e in log
        ]
        log_ph.markdown(
            "| Time | Event | Person |\n|------|-------|--------|\n" + "\n".join(rows)
        )

    fps_ph.metric("Recognition FPS", f"{fps:.1f}")
    mic_ph.metric("Microphone", mic_status)

    time.sleep(0.05)
