"""Diarization pipeline: face detection + identity + speaking detection.

Processes a video file and produces a speaking log with per-segment identity
labels. Accepts a pluggable IdentityModule — defaults to NullIdentity (no
face recognition, track-ID-only labels).
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_AR_ROOT = Path(__file__).resolve().parent.parent
if str(_AR_ROOT) not in sys.path:
    sys.path.insert(0, str(_AR_ROOT))

from config import SAMPLE_RATE, CAMERA_FPS
from models import IdentityModule
from pipeline.identity import NullIdentity, FullIdentity
from processing.face_detector import FaceDetector
from processing.face_tracker import FaceTracker
from processing.face_embedder import FaceEmbedder
from processing.face_matcher import FaceMatcher
from processing.speaking_detector import SpeakingDetector
from storage.speaking_log import SpeakingLog
from storage.database import Database


class DiarizationPipeline:
    def __init__(self, identity: IdentityModule | None = None):
        self._identity = identity or NullIdentity()

    def run(self, video_path: Path) -> list[dict]:
        audio = _extract_audio(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or CAMERA_FPS
        detector = FaceDetector()
        tracker = FaceTracker()
        speaker = SpeakingDetector(use_mic=False)
        log = SpeakingLog()

        speaker.feed_audio(audio)

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = frame_idx / fps
                faces = detector.detect(frame, timestamp=timestamp)

                raw_matches = [self._identity.identify(face, frame_idx) for face in faces]
                smoothed, track_ids = tracker.update(faces, raw_matches, frame_idx)

                active_ids = set(track_ids)
                for face, tid in zip(faces, track_ids):
                    speaker.add_crop(tid, face.crop)

                speaker.run_inference(frame_idx, active_track_ids=active_ids)

                for face, match, tid in zip(faces, smoothed, track_ids):
                    is_speaking = speaker.get_speaking(tid)
                    log.update(
                        track_id=tid,
                        person_id=match.person_id if match.is_known else None,
                        name=match.name if match.is_known else f"track_{tid}",
                        is_speaking=is_speaking,
                        timestamp=timestamp,
                    )

                frame_idx += 1
        finally:
            cap.release()
            speaker.close()
            detector.close()

        log.close(timestamp=frame_idx / fps)
        return log.get_segments()


def _extract_audio(video_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
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


if __name__ == "__main__":
    VIDEO_PATH = _AR_ROOT.parent / "test_movie.mp4"

    if not VIDEO_PATH.exists():
        print(f"Test video not found: {VIDEO_PATH}")
        sys.exit(1)

    db = Database()
    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher = FaceMatcher()

    pipeline = DiarizationPipeline(identity=FullIdentity(embedder, matcher, db))
    segments = pipeline.run(VIDEO_PATH)

    print(f"\n{'='*60}")
    print(f"Speaking log: {len(segments)} segments")
    print(f"{'='*60}")

    track_ids = set()
    for seg in segments:
        print(seg)
        track_ids.add(seg["track_id"])
        dur = seg["end"] - seg["start"]
        print(f"  track={seg['track_id']:2d}  {seg['start']:7.2f}s - {seg['end']:7.2f}s  ({dur:.2f}s)  {seg['name']}")

    print(f"\nDistinct tracks: {sorted(track_ids)}")
    print(f"Total segments:  {len(segments)}")
