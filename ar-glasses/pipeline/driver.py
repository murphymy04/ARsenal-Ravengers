"""Pipeline driver: single-pass video extraction feeding both pipelines.

Extracts frames and audio from a video file once, then passes them into
the diarization and transcription pipelines. This removes redundant I/O
and sets up the architecture for live video later.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_AR_ROOT = Path(__file__).resolve().parent.parent
if str(_AR_ROOT) not in sys.path:
    sys.path.insert(0, str(_AR_ROOT))

from config import SAMPLE_RATE, CAMERA_FPS
from models import TranscriptSegment
from pipeline.diarization import DiarizationPipeline
from pipeline.transcription import TranscriptionPipeline

_DIARIZATION_CACHE = _AR_ROOT / "data" / "diarization_cache.json"


def _extract_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or CAMERA_FPS
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def _extract_audio_pcm(video_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
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

def _extract_audio_wav(video_path: Path) -> bytes:
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav",
        "-loglevel", "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout

class PipelineDriver:
    def __init__(
        self,
        diarization: DiarizationPipeline,
        transcription: TranscriptionPipeline,
    ):
        self._diarization = diarization
        self._transcription = transcription

    def run(self, video_path: Path) -> tuple[list[dict], list[TranscriptSegment]]:
        if _DIARIZATION_CACHE.exists():
            print(f"Loading diarization from cache: {_DIARIZATION_CACHE}")
            with open(_DIARIZATION_CACHE) as f:
                diarization_segments = json.load(f)
        else:
            frames, fps = _extract_frames(video_path)
            audio_pcm = _extract_audio_pcm(video_path)
            diarization_segments = self._diarization.run(frames, fps, audio_pcm)
            _DIARIZATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(_DIARIZATION_CACHE, "w") as f:
                json.dump(diarization_segments, f, indent=2)

        audio_wav = _extract_audio_wav(video_path)
        transcript_segments = self._transcription.run(audio_wav)

        return diarization_segments, transcript_segments


def combine_segments(
    diarization_segments: list[dict],
    transcript_segments: list[TranscriptSegment],
) -> list[dict]:
    result = []
    for seg in transcript_segments:
        best_name, best_overlap = "wearer", 1.5
        for sp in diarization_segments:
            overlap = min(seg.end_time, sp["end"]) - max(seg.start_time, sp["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = sp["name"]
        result.append({
            "speaker": best_name,
            "text": seg.text,
            "start": seg.start_time,
            "end": seg.end_time,
        })
    return result

if __name__ == "__main__":
    from pipeline.identity import NullIdentity, FullIdentity
    from processing.face_detector import FaceDetector
    from processing.face_embedder import FaceEmbedder
    from processing.face_matcher import FaceMatcher
    from storage.database import Database

    VIDEO_PATH = _AR_ROOT.parent / "test_movie.mp4"

    if not VIDEO_PATH.exists():
        print(f"Test video not found: {VIDEO_PATH}")
        sys.exit(1)

    db = Database()
    embedder = FaceEmbedder()
    matcher = FaceMatcher()

    driver = PipelineDriver(
        diarization=DiarizationPipeline(identity=FullIdentity(embedder, matcher, db)),
        transcription=TranscriptionPipeline(),
    )

    diarization_segments, transcript_segments = driver.run(VIDEO_PATH)

    DiarizationPipeline.test(diarization_segments)
    TranscriptionPipeline.test(transcript_segments)

    combined = combine_segments(diarization_segments, transcript_segments)
    print(f"\n{'='*60}")
    print(f"Combined: {len(combined)} segments")
    print(f"{'='*60}")
    for seg in combined:
        print(f"  [{seg['start']:7.2f} - {seg['end']:7.2f}] {seg['speaker']}: {seg['text']}")
