"""Flask debug dashboard for the diarization + retrieval pipeline.

Streams:
  * annotated MJPEG frames with face boxes, identity labels, speaking rings
  * flushed conversations (speaker + single transcript paragraph as sent to Zep)
    with a toggle to fall back to segment-by-segment diarization
  * a retrieval panel populated from driver.retrieval_results
  * a recording indicator (idle / recording / flushing)
  * a live RMS graph of the incoming audio stream (wallclock-paced)

Default mode serves audio via a browser <audio> tag and throttles the
processing loop to video FPS for best-effort A/V sync. --fast skips
audio, skips the throttle, and uses VISION_STRIDE. --glasses also
skips the audio element (live glasses mode produces no playback track).

Usage:
  python dashboard.py path/to/video.mp4
  python dashboard.py --fast path/to/video.mp4
  python dashboard.py --glasses
  python dashboard.py --no-identity path/to/video.mp4
  python dashboard.py --host 0.0.0.0 --port 5050 path/to/video.mp4

Flags:
  video_path      Positional path to the video file (omit with --glasses).
  --glasses       Run against a live glasses stream instead of a file.
  --no-identity   Disable the EdgeFace identity module (faster, no names).
  --fast          Skip audio + throttle + apply VISION_STRIDE.
  --host / --port Bind address for the Flask server.
"""

import argparse
import logging
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from flask_socketio import SocketIO

from config import (
    CAMERA_FPS,
    HUD_BROADCAST_ENABLED,
    HUD_BROADCAST_HOST,
    HUD_BROADCAST_PORT,
    LIVE_BUFFER_SECONDS,
    RETRIEVAL_COOLDOWN_SECONDS,
    RETRIEVAL_ENABLED,
    SAMPLE_RATE,
    SIMULATION_AUDIO_GAIN,
    VISION_STRIDE,
)
from input.camera import Camera
from input.microphone import SimulatedMic
from pipeline.diarization import DiarizationPipeline
from pipeline.identity import FullIdentity, NullIdentity
from pipeline.live import (
    LivePipelineDriver,
    extract_audio_pcm,
    get_video_fps,
    pcm_to_wav,
)
from pipeline.recording_buffer import ChunkData, sanitize
from pipeline.transcription import TranscriptionPipeline
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from storage.database import Database

try:
    from pipeline.retrieval import RetrievalWorker
except ImportError:
    RetrievalWorker = None

try:
    from pipeline.hud_broadcast import HudBroadcastServer
except ImportError:
    HudBroadcastServer = None

try:
    from pipeline.knowledge import flush_memory
except ImportError:
    flush_memory = None

logging.getLogger("werkzeug").setLevel(logging.ERROR)


class RmsLiveGraph:
    """Paces incoming audio chunks to wallclock and emits RMS points over SocketIO.

    push_audio() is cheap: enqueue only. A background thread drains the queue
    and emits one point per chunk, delayed so that the stream of emitted
    points advances at 1s per wallclock second regardless of how fast the
    producer runs.
    """

    def __init__(self, socketio: SocketIO, sample_rate: int):
        self._socketio = socketio
        self._sample_rate = sample_rate
        self._queue: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drip_loop, daemon=True)
        self._thread.start()

    def push_audio(self, chunk: np.ndarray, video_time: float) -> None:
        self._queue.put((chunk.copy(), float(video_time)))

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)

    def _drip_loop(self) -> None:
        wall_start: float | None = None
        video_start: float | None = None
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                return
            chunk, video_time = item

            if wall_start is None:
                wall_start = time.perf_counter()
                video_start = video_time

            target_wall = wall_start + (video_time - video_start)
            delay = target_wall - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

            rms = float(np.sqrt(np.mean(np.square(chunk.astype(np.float32)))))
            if not math.isfinite(rms):
                rms = 0.0

            self._socketio.emit("rms", {"t": video_time - video_start, "rms": rms})


@dataclass
class DebugState:
    latest_jpeg: bytes | None = None
    audio_wav: bytes | None = None
    captions: list[dict] = field(default_factory=list)
    conversations: list[dict] = field(default_factory=list)
    retrieval: list[dict] = field(default_factory=list)
    video_fps: float = 30.0
    video_time: float = 0.0
    effective_rate: float = 0.0
    recording: bool = False
    flushing: bool = False
    finished: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def draw_overlay(frame, faces, matches, track_ids, speaker, timestamp, frame_idx):
    out = frame.copy()

    for face, match, tid in zip(faces, matches, track_ids, strict=False):
        b = face.bbox
        is_speaking = speaker.get_speaking(tid)
        color = (0, 220, 60) if match.is_known else (30, 80, 220)

        overlay = out.copy()
        cv2.rectangle(overlay, (b.x1, b.y1), (b.x2, b.y2), color, -1)
        cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), color, 4 if is_speaking else 2)

        if is_speaking:
            cx = (b.x1 + b.x2) // 2
            cv2.circle(out, (cx, b.y1 - 18), 14, (0, 255, 255), 2)
            cv2.putText(
                out,
                "SPK",
                (cx - 14, b.y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

        label = f"{match.name} · t{tid}"
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

    header = f"{timestamp:.2f}s  frame {frame_idx}"
    cv2.putText(
        out, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    return out


def encode_jpeg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def build_identity(use_identity: bool):
    if not use_identity:
        return None, NullIdentity()
    db = Database()
    return db, FullIdentity(FaceEmbedder(), FaceMatcher(), db)


def drain_new_retrieval(
    driver: LivePipelineDriver, last_count: int
) -> tuple[list[dict], int]:
    new_items = driver.retrieval_results[last_count:]
    rendered = [
        {
            "name": name,
            "person_id": person_id,
            "last_spoke": context.get("last_spoke"),
            "last_spoke_about": context.get("last_spoke_about"),
            "ask_about": context.get("ask_about"),
            "raw_facts": list(context.get("raw_facts") or []),
        }
        for name, person_id, context in new_items
    ]
    return rendered, len(driver.retrieval_results)


def install_flush_interceptor(driver: LivePipelineDriver, state: DebugState):
    def flushing_sanitize_and_flush(chunks: list[ChunkData]):
        with state.lock:
            state.flushing = True
        try:
            conversation = sanitize(chunks)
            if not conversation.transcript:
                return

            driver._store_interaction(conversation.person_id, conversation.transcript)
            with state.lock:
                state.conversations.append(
                    {
                        "spoke_with": conversation.spoke_with,
                        "person_id": conversation.person_id,
                        "transcript": conversation.transcript,
                        "start": conversation.window_start,
                        "end": conversation.window_end,
                    }
                )

            print(
                f"\n## RECORDING FLUSH [{conversation.window_start:.1f}s - "
                f"{conversation.window_end:.1f}s] spoke_with={conversation.spoke_with}"
            )
            print(f"  {conversation.transcript}")

            if driver.save_to_memory:
                driver._save_conversation(conversation)
        finally:
            with state.lock:
                state.flushing = False

    driver._sanitize_and_flush = flushing_sanitize_and_flush


def process_video(
    video_path: Path | None,
    use_identity: bool,
    fast: bool,
    state: DebugState,
    rms_graph: RmsLiveGraph,
    glasses: bool = False,
):
    glasses_server = None
    if glasses:
        from input.glasses_adapter import GlassesServer

        glasses_server = GlassesServer(sample_rate=SAMPLE_RATE)
        camera, mic, _ = glasses_server.start()
        fps = CAMERA_FPS
        with state.lock:
            state.video_fps = fps
    else:
        fps = get_video_fps(video_path)
        audio = extract_audio_pcm(video_path)

        with state.lock:
            state.audio_wav = pcm_to_wav(audio)
            state.video_fps = fps

        mic = SimulatedMic(audio, fps, gain=SIMULATION_AUDIO_GAIN, denoise=False)
        camera = Camera(source=str(video_path))

    db, identity = build_identity(use_identity)
    transcription = TranscriptionPipeline()
    driver = LivePipelineDriver(identity, transcription, db)
    install_flush_interceptor(driver, state)

    hud_server = None
    if HUD_BROADCAST_ENABLED:
        if HudBroadcastServer is None:
            raise RuntimeError(
                "HUD broadcast enabled but pipeline.hud_broadcast is unavailable."
            )
        hud_server = HudBroadcastServer(HUD_BROADCAST_HOST, HUD_BROADCAST_PORT)
        hud_server.start()
        driver._hud_server = hud_server

    track_event_queue = None
    retrieval_worker = None
    if RETRIEVAL_ENABLED and RetrievalWorker is not None:
        track_event_queue = queue.Queue()
        driver._retrieval_result_queue = queue.Queue()
        retrieval_worker = RetrievalWorker(
            track_event_queue,
            driver._retrieval_result_queue,
            RETRIEVAL_COOLDOWN_SECONDS,
        )
        retrieval_worker.start()

    diarization = DiarizationPipeline(
        identity=identity, track_event_queue=track_event_queue
    )
    diarization.open(fps)

    driver.start_flush_worker()

    if glasses:
        vision_stride = 1
    elif fast:
        vision_stride = VISION_STRIDE
    else:
        vision_stride = 2

    frame_idx = 0
    window_start = 0.0
    retrieval_seen = 0
    rate_samples: deque[tuple[float, float]] = deque(maxlen=60)

    try:
        for frame in camera.frames():
            timestamp = camera.last_timestamp_seconds if glasses else frame_idx / fps
            chunk = mic.advance_frame()
            rms_graph.push_audio(chunk, timestamp)
            is_vision_frame = frame_idx % vision_stride == 0

            diarization.process_frame(
                frame, chunk, frame_idx, timestamp, is_vision_frame
            )

            annotated = draw_overlay(
                frame,
                diarization._last_faces,
                diarization._last_smoothed,
                diarization._last_track_ids,
                diarization._speaker,
                timestamp,
                frame_idx,
            )
            jpeg = encode_jpeg(annotated)

            rate_samples.append((time.perf_counter(), timestamp))
            effective_rate = 0.0
            if len(rate_samples) >= 2:
                wall_span = rate_samples[-1][0] - rate_samples[0][0]
                video_span = rate_samples[-1][1] - rate_samples[0][1]
                if wall_span > 0:
                    effective_rate = video_span / wall_span

            with state.lock:
                state.latest_jpeg = jpeg
                state.video_time = timestamp
                state.effective_rate = effective_rate
                state.recording = driver.recording_buffer.flag

            driver._publish_retrieval_results()
            new_retrieval, retrieval_seen = drain_new_retrieval(driver, retrieval_seen)
            if new_retrieval:
                with state.lock:
                    state.retrieval.extend(new_retrieval)

            completed = driver.drain_flush_results()
            if completed:
                with state.lock:
                    for combined, _ in completed:
                        state.captions.extend(combined)

            if timestamp - window_start >= LIVE_BUFFER_SECONDS:
                driver.submit_flush(diarization, mic, window_start, timestamp)
                window_start = timestamp

            frame_idx += 1

        final_end = camera.last_timestamp_seconds if glasses else frame_idx / fps
        if final_end > window_start:
            driver.submit_flush(diarization, mic, window_start, final_end)
    finally:
        driver.stop_flush_worker()
        completed = driver.drain_flush_results()
        new_retrieval, retrieval_seen = drain_new_retrieval(driver, retrieval_seen)
        with state.lock:
            for combined, _ in completed:
                state.captions.extend(combined)
            state.retrieval.extend(new_retrieval)

        remaining = driver.recording_buffer.drain()
        if remaining:
            driver._sanitize_and_flush(remaining)
        with state.lock:
            state.recording = False
        if driver.save_to_memory and flush_memory:
            print("[knowledge] waiting for pending saves...")
            flush_memory()
            print("[knowledge] done.")
        if retrieval_worker:
            retrieval_worker.stop()
        if hud_server:
            hud_server.stop()
        diarization.close()
        camera.close()
        if glasses_server:
            glasses_server.stop()
        with state.lock:
            state.finished = True
        print("[dashboard] processing finished")


def build_app(state: DebugState, fast: bool, glasses: bool) -> tuple[Flask, SocketIO]:
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
    hide_audio = fast or glasses

    @app.route("/")
    def index():
        return render_template("dashboard.html", hide_audio=hide_audio)

    @app.route("/video")
    def video():
        def stream():
            boundary = b"--frame"
            last_jpeg_id = 0
            while True:
                with state.lock:
                    jpeg = state.latest_jpeg
                    finished = state.finished
                if jpeg is not None:
                    jpeg_id = id(jpeg)
                    if jpeg_id != last_jpeg_id:
                        last_jpeg_id = jpeg_id
                        yield (
                            boundary
                            + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                if finished and jpeg is not None:
                    break
                time.sleep(1.0 / 60)

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/audio")
    def audio():
        with state.lock:
            wav = state.audio_wav
        if wav is None:
            return ("", 404)
        return Response(wav, mimetype="audio/wav")

    @app.route("/state.json")
    def state_json():
        with state.lock:
            return jsonify(
                captions=state.captions,
                conversations=state.conversations,
                retrieval=state.retrieval,
                finished=state.finished,
                video_fps=state.video_fps,
                video_time=state.video_time,
                effective_rate=state.effective_rate,
                recording=state.recording,
                flushing=state.flushing,
            )

    return app, socketio


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video_path", type=Path, nargs="?", help="Video file to play.")
    parser.add_argument(
        "--glasses", action="store_true", help="Use live glasses stream."
    )
    parser.add_argument(
        "--no-identity", action="store_true", help="Disable EdgeFace identity."
    )
    parser.add_argument(
        "--fast", action="store_true", help="Skip audio + throttle; use VISION_STRIDE."
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)."
    )
    parser.add_argument(
        "--port", type=int, default=5050, help="Bind port (default 5050)."
    )
    args = parser.parse_args()

    if not args.glasses:
        if args.video_path is None:
            parser.error("video_path is required unless --glasses is set")
        if not args.video_path.exists():
            print(f"Video not found: {args.video_path}")
            sys.exit(1)

    state = DebugState()
    app, socketio = build_app(state, args.fast, args.glasses)
    rms_graph = RmsLiveGraph(socketio, sample_rate=SAMPLE_RATE)

    worker = threading.Thread(
        target=process_video,
        args=(args.video_path, not args.no_identity, args.fast, state, rms_graph),
        kwargs={"glasses": args.glasses},
        daemon=True,
    )
    worker.start()

    print(f"Dashboard: http://{args.host}:{args.port}")
    socketio.run(
        app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True
    )


if __name__ == "__main__":
    main()
