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

from models import TranscriptSegment
from pipeline.conversation_end import is_conversation_end
from pipeline.diarization import DiarizationPipeline
from pipeline.identity import FullIdentity
from pipeline.transcription import TranscriptionPipeline
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from storage.database import Database

from config import CAMERA_FPS, SAMPLE_RATE

_DIARIZATION_CACHE_DIR = _AR_ROOT / "data" / "diarization_cache"


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


def _extract_audio_wav(video_path: Path) -> bytes:
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        "-loglevel",
        "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout


class PipelineDriver:
    def __init__(
        self,
        diarization: DiarizationPipeline,
        transcription=None,
    ):
        self._diarization = diarization
        self._transcription = transcription

    def run(self, video_path: Path) -> tuple[list[dict], list[TranscriptSegment]]:
        _DIARIZATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _DIARIZATION_CACHE_DIR / f"{video_path.stem}.json"

        if cache_file.exists():
            print(f"Loading diarization from cache: {cache_file.name}")

            with open(cache_file) as f:
                diarization_segments = json.load(f)
        else:
            frames, fps = _extract_frames(video_path)
            audio_pcm = _extract_audio_pcm(video_path)

            diarization_segments = self._diarization.run(frames, fps, audio_pcm)

            with open(cache_file, "w") as f:
                json.dump(diarization_segments, f, indent=2)

        if self._transcription:
            audio_wav = _extract_audio_wav(video_path)
            transcript_segments = self._transcription.run(audio_wav)
        else:
            transcript_segments = []

        return diarization_segments, transcript_segments


def combine_segments(
    diarization_segments: list[dict],
    transcript_segments: list[TranscriptSegment],
    min_coverage: float = 0.4,
) -> list[dict]:
    result = []
    for seg in transcript_segments:
        best_name, best_person_id, best_coverage = "wearer", None, 0.0
        seg_duration = seg.end_time - seg.start_time

        for sp in diarization_segments:
            overlap = min(seg.end_time, sp["end"]) - max(seg.start_time, sp["start"])

            if overlap <= 0:
                continue

            coverage = overlap / seg_duration
            if coverage > best_coverage:
                best_coverage = coverage
                best_name = sp["name"]
                best_person_id = sp["person_id"]

        if best_coverage < min_coverage:
            best_name = "wearer"
            best_person_id = None

        result.append(
            {
                "speaker": best_name,
                "person_id": best_person_id,
                "text": seg.text,
                "start": seg.start_time,
                "end": seg.end_time,
            }
        )

    return result


def split_into_conversations(
    combined: list[dict],
    chunk_seconds: float = 10.0,
) -> list[list[dict]]:
    if not combined:
        return []

    conversations: list[list[dict]] = []
    current: list[dict] = []
    chunk_start = combined[0]["start"]

    for seg in combined:
        current.append(seg)

        if seg["end"] - chunk_start >= chunk_seconds:
            chunk = [s for s in current if s["start"] >= chunk_start]
            if is_conversation_end(chunk):
                conversations.append(current)
                current = []
            chunk_start = seg["end"]

    if current:
        conversations.append(current)

    return conversations


def _collect_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.mp4")) + sorted(path.glob("*.MP4"))


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else _AR_ROOT / "test_videos"
    videos = _collect_videos(target)

    if not videos:
        print(f"No videos found at: {target}")
        sys.exit(1)

    db = Database()
    driver = PipelineDriver(
        diarization=DiarizationPipeline(
            identity=FullIdentity(FaceEmbedder(), FaceMatcher(), db)
        ),
        transcription=TranscriptionPipeline(),
    )

    for video_path in videos:
        print(f"\n{'=' * 60}")
        print(f"Processing: {video_path.name}")
        print(f"{'=' * 60}")

        diarization_segments, transcript_segments = driver.run(video_path)
        combined = combine_segments(diarization_segments, transcript_segments)
        conversations = split_into_conversations(combined)

        for i, conv in enumerate(conversations):
            print(f"\n  Conversation {i + 1} ({len(conv)} segments)")
            for seg in conv:
                print(
                    f"    [{seg['start']:7.2f} - {seg['end']:7.2f}] "
                    f"{seg['speaker']}: {seg['text']}"
                )
            print(f"  --- end of conversation {i + 1} ---")
