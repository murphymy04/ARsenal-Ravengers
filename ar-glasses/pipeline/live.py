"""Live streaming pipeline driver.

Captures camera frames and mic audio continuously, running diarization
frame-by-frame and transcription in buffered windows.

Supports both live camera/mic and offline video simulation through
swappable audio source (Microphone vs SimulatedMic) and clock function
(wall-clock vs frame-rate-based).
"""

import io
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import CAMERA_FPS, LIVE_BUFFER_SECONDS, SAMPLE_RATE, SAVE_TO_MEMORY, SIMULATION_AUDIO_GAIN, SPEAKING_BACKEND
from input.camera import Camera
from input.microphone import Microphone
from models import IdentityModule
from pipeline.driver import combine_segments
from pipeline.segments import merge_close_segments
from pipeline.transcription import TranscriptionPipeline
from processing.face_detector import FaceDetector
from processing.face_tracker import FaceTracker
from storage.speaking_log import SpeakingLog


def _create_speaker(fps: float):
    if SPEAKING_BACKEND == "vad_rms":
        from processing.vad_speaker import VadSpeaker
        return VadSpeaker(fps=fps)
    from processing.speaking_detector import SpeakingDetector
    return SpeakingDetector(fps=fps)


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
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-loglevel", "error",
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
    def __init__(self, identity: IdentityModule, transcription: TranscriptionPipeline):
        self.identity = identity
        self.transcription = transcription
        self.save_to_memory = SAVE_TO_MEMORY
        self.conversation_buffer: list[dict] = []

    def run(self, camera: Camera, mic=None, clock_fn=None, fps: float = CAMERA_FPS) -> list[dict]:
        owns_mic = mic is None
        if owns_mic:
            mic = Microphone()
            mic.open()

        if clock_fn is None:
            stream_start = time.time()
            clock_fn = lambda frame_idx: time.time() - stream_start

        detector = FaceDetector()
        tracker = FaceTracker()
        speaker = _create_speaker(fps)
        log = SpeakingLog()

        all_combined: list[dict] = []
        all_diarization: list[dict] = []
        window_start = 0.0
        frame_idx = 0
        window_face_counts: list[int] = []
        window_speaking_frames = 0

        try:
            for frame in camera.frames():
                timestamp = clock_fn(frame_idx)

                chunk = mic.advance_frame()
                speaker.drip_audio(chunk)

                faces = detector.detect(frame, timestamp=timestamp)
                raw_matches = [self.identity.identify(face, frame_idx) for face in faces]
                smoothed, track_ids = tracker.update(faces, raw_matches, frame_idx)

                for face, tid in zip(faces, track_ids):
                    speaker.add_crop(tid, face.crop)
                speaker.run_inference(frame_idx, active_track_ids=set(track_ids))

                window_face_counts.append(len(faces))
                for face, match, tid in zip(faces, smoothed, track_ids):
                    is_speaking = speaker.get_speaking(tid)
                    if is_speaking:
                        window_speaking_frames += 1
                    log.update(
                        track_id=tid,
                        person_id=match.person_id if match.is_known else None,
                        name=match.name if match.is_known else f"track_{tid}",
                        is_speaking=is_speaking,
                        timestamp=timestamp,
                    )

                frame_idx += 1

                if timestamp - window_start >= LIVE_BUFFER_SECONDS:
                    n_frames = len(window_face_counts)
                    avg_faces = sum(window_face_counts) / n_frames if n_frames else 0
                    print(f"\n  [debug] {n_frames} frames, avg {avg_faces:.1f} faces/frame, {window_speaking_frames} speaking-true frames")
                    combined, diarization = self.flush_window(log, mic, window_start, timestamp)
                    all_combined.extend(combined)
                    all_diarization.extend(diarization)
                    log = SpeakingLog()
                    window_start = timestamp
                    window_face_counts = []
                    window_speaking_frames = 0

        finally:
            if self.save_to_memory and self.conversation_buffer:
                from pipeline.knowledge import save_to_memory
                save_to_memory(self.conversation_buffer)
            speaker.close()
            detector.close()
            if owns_mic:
                mic.close()

        return all_combined, all_diarization

    def flush_window(self, log, mic, window_start, window_end) -> tuple[list[dict], list[dict]]:
        log.close(timestamp=window_end)
        diarization_segments = merge_close_segments(log.get_segments())

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

        if self.save_to_memory and combined:
            self.conversation_buffer.extend(combined)
            from pipeline.conversation_end import is_conversation_end
            if is_conversation_end(combined):
                from pipeline.knowledge import save_to_memory
                save_to_memory(self.conversation_buffer)
                self.conversation_buffer = []

        print(f"\n{'='*60}")
        print(f"[{window_start:.1f}s - {window_end:.1f}s]")
        print(f"{'='*60}")
        print(f"  ASD segments ({len(diarization_segments)}):")
        for seg in diarization_segments:
            print(f"    [{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['name']} (track {seg['track_id']})")
        print(f"  Combined ({len(combined)}):")
        for seg in combined:
            print(f"    [{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['speaker']}: {seg['text']}")

        return combined, diarization_segments


if __name__ == "__main__":
    from pipeline.identity import FullIdentity
    from processing.face_embedder import FaceEmbedder
    from processing.face_matcher import FaceMatcher
    from storage.database import Database

    AR_ROOT = Path(__file__).resolve().parent.parent
    SIMULATION_CACHE_DIR = AR_ROOT / "data" / "simulation_cache"

    db = Database()
    identity = FullIdentity(FaceEmbedder(), FaceMatcher(), db)
    transcription = TranscriptionPipeline()
    driver = LivePipelineDriver(identity, transcription)

    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            sys.exit(1)

        SIMULATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = SIMULATION_CACHE_DIR / f"{video_path.stem}.json"

        if cache_file.exists():
            print(f"Loading simulation from cache: {cache_file.name}")
            with open(cache_file) as f:
                cached = json.load(f)
            combined = cached["combined"]
            diarization = cached["diarization"]
            print(f"\n  ASD segments ({len(diarization)}):")
            for seg in diarization:
                print(f"    [{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['name']} (track {seg['track_id']})")
            print(f"\n  Combined ({len(combined)}):")
            for seg in combined:
                print(f"  [{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['speaker']}: {seg['text']}")
        else:
            from input.microphone import SimulatedMic

            fps = get_video_fps(video_path)
            print(f"Simulating: {video_path.name} ({fps:.1f} fps)")

            audio = extract_audio_pcm(video_path)
            sim_mic = SimulatedMic(audio, fps, gain=SIMULATION_AUDIO_GAIN, denoise=True)
            camera = Camera(source=str(video_path))

            combined, diarization = driver.run(
                camera,
                mic=sim_mic,
                clock_fn=lambda fi, f=fps: fi / f,
                fps=fps,
            )

            with open(cache_file, "w") as f:
                json.dump({"combined": combined, "diarization": diarization}, f, indent=2)
            print(f"Saved simulation cache: {cache_file.name}")
    else:
        camera = Camera()
        driver.run(camera)
