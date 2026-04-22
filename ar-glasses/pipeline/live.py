"""Live streaming pipeline driver.

Captures camera frames and mic audio continuously, running diarization
frame-by-frame and transcription in buffered windows.

Supports both live camera/mic and offline video simulation through
swappable audio source (Microphone vs SimulatedMic) and clock function
(wall-clock vs frame-rate-based).
"""

import argparse
import json
import queue
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from input.camera import Camera
from input.microphone import Microphone, SimulatedMic
from models import IdentityModule
from pipeline.diarization import DiarizationPipeline
from pipeline.flush_worker import FlushWorker
from pipeline.identity import FullIdentity
from pipeline.recording_buffer import (
    ChunkData,
    RecordingBuffer,
    SanitizedConversation,
    sanitize,
)
from pipeline.transcription import TranscriptionPipeline
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from storage.database import Database

from config import (
    CAMERA_FPS,
    HUD_BROADCAST_ENABLED,
    HUD_BROADCAST_HOST,
    HUD_BROADCAST_PORT,
    LIVE_BUFFER_SECONDS,
    RETRIEVAL_COOLDOWN_SECONDS,
    RETRIEVAL_ENABLED,
    SAMPLE_RATE,
    SAVE_TO_MEMORY,
    SIMULATION_AUDIO_GAIN,
    VISION_STRIDE,
)

try:
    from pipeline.knowledge import flush_memory, save_to_memory
except ImportError:
    save_to_memory = None
    flush_memory = None

try:
    from pipeline.retrieval import RetrievalWorker, drain_results
except ImportError:
    RetrievalWorker = None
    drain_results = None

try:
    from pipeline.hud_broadcast import HudBroadcastServer
except ImportError:
    HudBroadcastServer = None


def extract_audio_pcm(video_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-loglevel",
        "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or CAMERA_FPS
    cap.release()
    return fps


class LivePipelineDriver:
    def __init__(
        self,
        identity: IdentityModule,
        transcription: TranscriptionPipeline,
        db=None,
    ):
        self.identity = identity
        self.transcription = transcription
        self._db = db
        self.save_to_memory = SAVE_TO_MEMORY
        self.recording_buffer = RecordingBuffer()
        self.retrieval_results: list[tuple] = []
        self._retrieval_result_queue = None
        self._hud_server = None
        self.flush_worker: FlushWorker | None = None

    def _save_conversation(self, conversation: SanitizedConversation):
        if conversation.person_id is None or not self._db:
            return
        person = self._db.get_person(conversation.person_id)
        if not person or not person.is_labeled:
            label = person.name if person else "unknown"
            print(f"  [knowledge] skipping Zep — {label} is not labeled")
            return

        if save_to_memory is None:
            raise RuntimeError(
                "SAVE_TO_MEMORY enabled but knowledge support unavailable."
            )
        save_to_memory(conversation.transcript, other_name=person.name)
        print(f"  [knowledge] flushed to Zep with resolved name: {person.name}")

    def _sanitize_and_flush(self, chunks: list[ChunkData]):
        conversation = sanitize(chunks)
        if not conversation.transcript:
            return

        self._store_interaction(conversation.person_id, conversation.transcript)

        print(
            f"\n## RECORDING FLUSH [{conversation.window_start:.1f}s - "
            f"{conversation.window_end:.1f}s] spoke_with={conversation.spoke_with}"
        )
        print(f"  {conversation.transcript}")

        if self.save_to_memory:
            self._save_conversation(conversation)

    def _publish_retrieval_results(self):
        if not self._retrieval_result_queue:
            return
        if drain_results is None:
            raise RuntimeError(
                "Retrieval result queue exists but drain_results is unavailable."
            )
        for result in drain_results(self._retrieval_result_queue):
            self.retrieval_results.append(result)
            person_name, person_id, context = result
            print(f"\n  [retrieval] {person_name}:")
            print(f"    last_spoke:       {context['last_spoke']}")
            print(f"    last_spoke_about: {context['last_spoke_about']}")
            print(f"    ask_about:        {context['ask_about']}")
            print(f"    raw_facts ({len(context['raw_facts'])}):")
            for fact in context["raw_facts"]:
                print(f"      - {fact}")

            if self._hud_server:
                print(f"  [hud] publishing person_context for {person_name}")
                hud_context = {
                    "type": "person_context",
                    "name": person_name,
                    "person_id": person_id,
                    "context": context,
                }
                print(hud_context)
                self._hud_server.publish(hud_context)
            else:
                print("  [hud] broadcast disabled — not sending")

    def _store_interaction(self, person_id: int | None, transcript: str):
        if not self._db:
            return
        self._db.add_interaction(person_id, transcript)

    def start_flush_worker(self):
        self.flush_worker = FlushWorker(
            transcription=self.transcription,
            recording_buffer=self.recording_buffer,
            sanitize_and_flush=lambda chunks: self._sanitize_and_flush(chunks),
        )
        self.flush_worker.start()

    def submit_flush(
        self,
        diarization: DiarizationPipeline,
        mic,
        window_start: float,
        window_end: float,
    ):
        diarization_segments = diarization.take_segments(window_end)
        audio = mic.get_buffer_and_clear()
        self.flush_worker.submit(diarization_segments, audio, window_start, window_end)

    def drain_flush_results(self) -> list[tuple[list[dict], list[dict]]]:
        if not self.flush_worker:
            return []
        return self.flush_worker.drain_results()

    def stop_flush_worker(self):
        if not self.flush_worker:
            return
        self.flush_worker.stop()
        self.flush_worker = None

    def run(
        self,
        camera: Camera,
        mic=None,
        clock_fn=None,
        fps: float = CAMERA_FPS,
        vision_stride: int = 1,
        static_boundary: float | None = None,
        external_vision: bool = False,
    ) -> list[dict]:
        owns_mic = mic is None
        if owns_mic:
            mic = Microphone()
            mic.open()

        if clock_fn is None:
            stream_start = time.time()

            def clock_fn(frame_idx):
                return time.time() - stream_start

        track_event_queue = None
        retrieval_worker = None

        if HUD_BROADCAST_ENABLED:
            if HudBroadcastServer is None:
                raise RuntimeError(
                    "HUD broadcast enabled but pipeline.hud_broadcast is unavailable."
                )
            self._hud_server = HudBroadcastServer(
                HUD_BROADCAST_HOST, HUD_BROADCAST_PORT
            )
            self._hud_server.start()

        if RETRIEVAL_ENABLED and not external_vision:
            if RetrievalWorker is None:
                raise RuntimeError(
                    "Retrieval is enabled but pipeline.retrieval is unavailable."
                )
            track_event_queue = queue.Queue()
            self._retrieval_result_queue = queue.Queue()
            retrieval_worker = RetrievalWorker(
                track_event_queue,
                self._retrieval_result_queue,
                RETRIEVAL_COOLDOWN_SECONDS,
            )
            retrieval_worker.start()

        diarization = DiarizationPipeline(
            identity=self.identity,
            track_event_queue=None if external_vision else track_event_queue,
            static_boundary=static_boundary,
        )
        diarization.open(fps, owns_vision=not external_vision)

        self.start_flush_worker()

        all_combined: list[dict] = []
        all_diarization: list[dict] = []
        window_start = 0.0
        frame_idx = 0
        frames_since_flush = 0
        timestamp = 0.0

        if vision_stride > 1:
            print(
                f"  [stride] vision every {vision_stride} frames "
                f"({fps / vision_stride:.0f} detections/s)"
            )

        try:
            for item in camera.frames():
                if isinstance(item, tuple):
                    frame, vision_result = item
                else:
                    frame, vision_result = item, None

                timestamp = clock_fn(frame_idx)

                chunk = mic.advance_frame()
                is_vision_frame = frame_idx % vision_stride == 0
                diarization.process_frame(
                    frame,
                    chunk,
                    frame_idx,
                    timestamp,
                    is_vision_frame,
                    vision_result=vision_result,
                )

                frame_idx += 1
                frames_since_flush += 1

                self._publish_retrieval_results()

                for combined, diar in self.drain_flush_results():
                    all_combined.extend(combined)
                    all_diarization.extend(diar)

                if timestamp - window_start >= LIVE_BUFFER_SECONDS:
                    self.submit_flush(diarization, mic, window_start, timestamp)
                    window_start = timestamp
                    frames_since_flush = 0

            if frames_since_flush:
                self.submit_flush(diarization, mic, window_start, timestamp)

        finally:
            self.stop_flush_worker()
            for combined, diar in self.drain_flush_results():
                all_combined.extend(combined)
                all_diarization.extend(diar)

            remaining = self.recording_buffer.drain()
            if remaining:
                self._sanitize_and_flush(remaining)
            if self.save_to_memory and flush_memory:
                print("[knowledge] waiting for pending saves...")
                flush_memory()
                print("[knowledge] done.")
            if retrieval_worker:
                retrieval_worker.stop()
            if self._hud_server:
                self._hud_server.stop()
            diarization.close()
            if owns_mic:
                mic.close()

        return all_combined, all_diarization


if __name__ == "__main__":
    AR_ROOT = Path(__file__).resolve().parent.parent
    SIMULATION_CACHE_DIR = AR_ROOT / "data" / "simulation_cache"

    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", help="video file path")
    parser.add_argument(
        "--glasses",
        action="store_true",
        help="stream from AR glasses instead of webcam/video",
    )
    parser.add_argument(
        "--boundary", type=float, default=None, help="VAD static RMS boundary"
    )
    args, _ = parser.parse_known_args()

    db = Database()
    identity = FullIdentity(FaceEmbedder(), FaceMatcher(), db)
    transcription = TranscriptionPipeline()
    driver = LivePipelineDriver(identity, transcription, db)

    if args.glasses:
        from input.glasses_adapter import GlassesServer
        from processing.face_detector import FaceDetector
        from processing.face_tracker import FaceTracker

        retrieval_event_queue = queue.Queue() if RETRIEVAL_ENABLED else None
        retrieval_enqueue = (
            retrieval_event_queue.put_nowait if retrieval_event_queue else None
        )
        server = GlassesServer(
            sample_rate=SAMPLE_RATE,
            detector=FaceDetector(),
            identity_module=identity,
            tracker=FaceTracker(),
            retrieval_enqueue=retrieval_enqueue,
        )
        camera, mic, clock_fn = server.start()

        if retrieval_event_queue is not None and RetrievalWorker is not None:
            driver._retrieval_result_queue = queue.Queue()
            preconfigured_worker = RetrievalWorker(
                retrieval_event_queue,
                driver._retrieval_result_queue,
                RETRIEVAL_COOLDOWN_SECONDS,
            )
            preconfigured_worker.start()
        else:
            preconfigured_worker = None

        try:
            driver.run(
                camera,
                mic=mic,
                clock_fn=clock_fn,
                fps=CAMERA_FPS,
                vision_stride=5,
                static_boundary=args.boundary,
                external_vision=True,
            )
        finally:
            if preconfigured_worker:
                preconfigured_worker.stop()
            server.stop()
    elif args.source:
        video_path = Path(args.source)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            sys.exit(1)

        SIMULATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = SIMULATION_CACHE_DIR / f"{video_path.stem}.json"

        fps = get_video_fps(video_path)
        print(f"Simulating: {video_path.name} ({fps:.1f} fps)")

        audio = extract_audio_pcm(video_path)
        sim_mic = SimulatedMic(audio, fps, gain=SIMULATION_AUDIO_GAIN, denoise=False)
        camera = Camera(source=str(video_path))

        combined, diarization = driver.run(
            camera,
            mic=sim_mic,
            clock_fn=lambda fi, f=fps: fi / f,
            fps=fps,
            vision_stride=VISION_STRIDE,
            static_boundary=args.boundary,
        )

        with open(cache_file, "w") as f:
            json.dump({"combined": combined, "diarization": diarization}, f, indent=2)
        print(f"Saved simulation cache: {cache_file.name}")
    else:
        camera = Camera()
        driver.run(camera, static_boundary=args.boundary)
