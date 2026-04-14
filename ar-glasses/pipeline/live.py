"""Live streaming pipeline driver.

Captures camera frames and mic audio continuously, running diarization
frame-by-frame and transcription in buffered windows.

Supports both live camera/mic and offline video simulation through
swappable audio source (Microphone vs SimulatedMic) and clock function
(wall-clock vs frame-rate-based).
"""

import argparse
import io
import json
import queue
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from input.camera import Camera
from input.microphone import Microphone, SimulatedMic
from models import IdentityModule
from pipeline.diarization import DiarizationPipeline
from pipeline.driver import combine_segments
from pipeline.identity import FullIdentity
from pipeline.recording_buffer import (
    ChunkData,
    LongTurn,
    RecordingBuffer,
    detect_long_turn,
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


def pcm_to_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


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

    def _dominant_person_id(self, segments: list[dict]) -> int | None:
        counts: dict[int, int] = {}
        for seg in segments:
            pid = seg.get("person_id")
            if pid is not None:
                counts[pid] = counts.get(pid, 0) + 1

        if not counts:
            return None

        total = sum(counts.values())
        viable = {pid: n for pid, n in counts.items() if n / total >= 0.1}
        return max(viable, key=viable.get) if viable else None

    def _resolve_and_save(self, segments: list[dict]):
        person_id = self._dominant_person_id(segments)

        if person_id is None or not self._db:
            self._store_interaction(None, segments)
            return
        person = self._db.get_person(person_id)
        if not person:
            self._store_interaction(person_id, segments)
            return

        self._store_interaction(person_id, segments)

        if not person.is_labeled:
            print(f"  [knowledge] skipping Zep — {person.name} is not labeled")
            return

        resolved = []
        for seg in segments:
            if seg.get("person_id") == person_id:
                resolved.append({**seg, "speaker": person.name})
            else:
                resolved.append(seg)

        if save_to_memory is None:
            raise RuntimeError(
                "SAVE_TO_MEMORY enabled but knowledge support unavailable."
            )
        wearer_name = next(
            (s["speaker"] for s in resolved if s.get("person_id") != person_id),
            "Wearer",
        )
        save_to_memory(resolved, wearer_name=wearer_name, other_name=person.name)
        print(f"  [knowledge] flushed to Zep with resolved name: {person.name}")

    def _sanitize_and_flush(self, chunks: list[ChunkData]):
        sanitized = sanitize(chunks)
        combined: list[dict] = []
        for chunk in sanitized:
            combined.extend(chunk.combined)
        if not combined:
            return
        window_start = sanitized[0].window_start
        window_end = sanitized[-1].window_end
        print(
            f"\n## RECORDING FLUSH [{window_start:.1f}s - {window_end:.1f}s] "
            f"{len(sanitized)} chunks, {len(combined)} segments"
        )
        for seg in combined:
            print(
                f"  [{seg['start']:7.2f} - {seg['end']:7.2f}] "
                f"{seg['speaker']}: {seg['text']}"
            )
        if self.save_to_memory:
            self._resolve_and_save(combined)

    def _log_chunk_decision(
        self,
        chunk: ChunkData,
        long_turn: LongTurn | None,
        was_recording: bool,
        flushable,
    ):
        is_recording = self.recording_buffer.flag
        turn_str = (
            f"long_turn={long_turn.name} ratio={long_turn.ratio:.2f}"
            if long_turn
            else "long_turn=none"
        )
        transition = ""
        if not was_recording and is_recording:
            transition = "  -> FLAG ON"
        elif was_recording and not is_recording:
            transition = f"  -> FLAG OFF (flushed {len(flushable or [])} chunks)"

        print(
            f"\n## CHUNK [{chunk.window_start:.1f}s - {chunk.window_end:.1f}s] "
            f"{turn_str} recording={is_recording} "
            f"quiet={self.recording_buffer.quiet_chunks}{transition}"
        )

    def _store_interaction(self, person_id: int | None, segments: list[dict]):
        if not self._db:
            return
        transcript = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)
        self._db.add_interaction(person_id, transcript)

    def run(
        self,
        camera: Camera,
        mic=None,
        clock_fn=None,
        fps: float = CAMERA_FPS,
        vision_stride: int = 1,
        static_boundary: float | None = None,
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

        if RETRIEVAL_ENABLED:
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
            track_event_queue=track_event_queue,
            static_boundary=static_boundary,
        )
        diarization.open(fps)

        all_combined: list[dict] = []
        all_diarization: list[dict] = []
        window_start = 0.0
        frame_idx = 0
        window_face_counts: list[int] = []
        window_speaking_frames = 0

        if vision_stride > 1:
            print(
                f"  [stride] vision every {vision_stride} frames "
                f"({fps / vision_stride:.0f} detections/s)"
            )

        try:
            for frame in camera.frames():
                timestamp = clock_fn(frame_idx)

                chunk = mic.advance_frame()
                is_vision_frame = frame_idx % vision_stride == 0
                face_count, speaking_count = diarization.process_frame(
                    frame, chunk, frame_idx, timestamp, is_vision_frame
                )

                window_face_counts.append(face_count)
                window_speaking_frames += speaking_count
                frame_idx += 1

                if timestamp - window_start >= LIVE_BUFFER_SECONDS:
                    n_frames = len(window_face_counts)
                    avg_faces = sum(window_face_counts) / n_frames if n_frames else 0
                    print(
                        f"\n  [debug] {n_frames} frames, "
                        f"avg {avg_faces:.1f} faces/frame, "
                        f"{window_speaking_frames} speaking-true frames"
                    )
                    combined, diarization_segs = self.flush_window(
                        diarization, mic, window_start, timestamp
                    )
                    all_combined.extend(combined)
                    all_diarization.extend(diarization_segs)
                    window_start = timestamp
                    window_face_counts = []
                    window_speaking_frames = 0

            if window_face_counts:
                n_frames = len(window_face_counts)
                avg_faces = sum(window_face_counts) / n_frames if n_frames else 0
                print(
                    f"\n  [debug] {n_frames} frames, "
                    f"avg {avg_faces:.1f} faces/frame, "
                    f"{window_speaking_frames} speaking-true frames"
                )
                combined, diarization_segs = self.flush_window(
                    diarization, mic, window_start, timestamp
                )
                all_combined.extend(combined)
                all_diarization.extend(diarization_segs)

        finally:
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

    def flush_window(
        self, diarization: DiarizationPipeline, mic, window_start, window_end
    ) -> tuple[list[dict], list[dict]]:
        diarization_segments = diarization.take_segments(window_end)

        audio = mic.get_buffer_and_clear()
        if len(audio) == 0:
            print(f"\n[{window_start:.1f}s - {window_end:.1f}s] No audio captured")
            return [], []

        wav_bytes = pcm_to_wav(audio)
        transcript_segments = self.transcription.run(wav_bytes)

        for seg in transcript_segments:
            seg.start_time += window_start
            seg.end_time += window_start

        combined = combine_segments(diarization_segments, transcript_segments)

        long_turn = detect_long_turn(diarization_segments, window_start, window_end)
        chunk = ChunkData(
            audio=audio,
            diarization_segments=diarization_segments,
            transcript_segments=transcript_segments,
            combined=combined,
            window_start=window_start,
            window_end=window_end,
        )
        was_recording = self.recording_buffer.flag
        flushable = self.recording_buffer.ingest(chunk, long_turn)
        self._log_chunk_decision(chunk, long_turn, was_recording, flushable)
        if flushable is not None:
            self._sanitize_and_flush(flushable)

        print(f"\n{'=' * 60}")
        print(f"[{window_start:.1f}s - {window_end:.1f}s]")
        print(f"{'=' * 60}")
        print(f"  ASD segments ({len(diarization_segments)}):")
        for seg in diarization_segments:
            print(
                f"    [{seg['start']:7.2f} - {seg['end']:7.2f}] "
                f"{seg['name']} (track {seg['track_id']})"
            )
        print(f"  Combined ({len(combined)}):")
        for seg in combined:
            print(
                f"    [{seg['start']:7.2f} - {seg['end']:7.2f}] "
                f"{seg['speaker']}: {seg['text']}"
            )

        if self._retrieval_result_queue:
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
                    self._hud_server.publish(
                        {
                            "type": "person_context",
                            "name": person_name,
                            "person_id": person_id,
                            "context": context,
                        }
                    )

        return combined, diarization_segments


if __name__ == "__main__":
    AR_ROOT = Path(__file__).resolve().parent.parent
    SIMULATION_CACHE_DIR = AR_ROOT / "data" / "simulation_cache"

    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", help="video file or '--glasses'")
    parser.add_argument(
        "--boundary", type=float, default=None, help="VAD static RMS boundary"
    )
    args, _ = parser.parse_known_args()

    db = Database()
    identity = FullIdentity(FaceEmbedder(), FaceMatcher(), db)
    transcription = TranscriptionPipeline()
    driver = LivePipelineDriver(identity, transcription, db)

    if args.source == "--glasses":
        from input.glasses_adapter import GlassesServer

        server = GlassesServer(sample_rate=SAMPLE_RATE)
        camera, mic, clock_fn = server.start()
        try:
            driver.run(
                camera,
                mic=mic,
                clock_fn=clock_fn,
                fps=CAMERA_FPS,
                vision_stride=VISION_STRIDE,
                static_boundary=args.boundary,
            )
        finally:
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
