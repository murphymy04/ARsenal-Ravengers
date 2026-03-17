"""Diarization pipeline: face detection + identity + speaking detection.

Processes pre-extracted frames and audio, producing a speaking log with
per-segment identity labels. Accepts a pluggable IdentityModule — defaults
to NullIdentity (no face recognition, track-ID-only labels).

Stateless with respect to I/O: the driver is responsible for extracting
frames and audio from the source.
"""

import sys
from pathlib import Path

import numpy as np

_AR_ROOT = Path(__file__).resolve().parent.parent
if str(_AR_ROOT) not in sys.path:
    sys.path.insert(0, str(_AR_ROOT))

from config import CAMERA_FPS
from models import IdentityModule
from pipeline.identity import NullIdentity
from processing.face_detector import FaceDetector
from processing.face_tracker import FaceTracker
from processing.speaking_detector import SpeakingDetector
from storage.speaking_log import SpeakingLog


class DiarizationPipeline:
    def __init__(self, identity: IdentityModule | None = None):
        self._identity = identity or NullIdentity()

    def run(
        self,
        frames: list[np.ndarray],
        fps: float,
        audio: np.ndarray,
    ) -> list[dict]:
        detector = FaceDetector()
        tracker = FaceTracker()
        speaker = SpeakingDetector(use_mic=False, fps=fps)
        log = SpeakingLog()

        speaker.feed_audio(audio)

        try:
            for frame_idx, frame in enumerate(frames):
                timestamp = frame_idx / fps
                faces = detector.detect(frame, timestamp=timestamp)

                raw_matches = [self._identity.identify(face, frame_idx) for face in faces]
                smoothed, track_ids = tracker.update(faces, raw_matches, frame_idx)

                for face, tid in zip(faces, track_ids):
                    speaker.add_crop(tid, face.crop)

                speaker.run_inference(frame_idx, timestamp=timestamp)

                for face, match, tid in zip(faces, smoothed, track_ids):
                    is_speaking = speaker.get_speaking(tid)
                    log.update(
                        track_id=tid,
                        person_id=match.person_id if match.is_known else None,
                        name=match.name if match.is_known else f"track_{tid}",
                        is_speaking=is_speaking,
                        timestamp=timestamp,
                    )
        finally:
            speaker.close()
            detector.close()

        log.close(timestamp=len(frames) / fps)
        return self._merge_close_segments(log.get_segments())

    @staticmethod
    def _merge_close_segments(
        segments: list[dict], max_gap: float = 2.0
    ) -> list[dict]:
        if not segments:
            return segments

        by_track: dict[int, list[dict]] = {}
        for seg in segments:
            by_track.setdefault(seg["track_id"], []).append(seg)

        merged: list[dict] = []
        for segs in by_track.values():
            segs.sort(key=lambda s: s["start"])
            current = dict(segs[0])
            for nxt in segs[1:]:
                if nxt["start"] - current["end"] < max_gap:
                    current["end"] = nxt["end"]
                    if nxt.get("person_id") is not None:
                        current["person_id"] = nxt["person_id"]
                        current["name"] = nxt["name"]
                else:
                    merged.append(current)
                    current = dict(nxt)
            merged.append(current)

        merged.sort(key=lambda s: s["start"])
        return merged

    @staticmethod
    def test(segments: list[dict]):
        print(f"\n{'='*60}")
        print(f"Speaking log: {len(segments)} segments")
        print(f"{'='*60}")

        track_ids = set()
        for seg in segments:
            track_ids.add(seg["track_id"])
            dur = seg["end"] - seg["start"]
            print(f"  track={seg['track_id']:2d}  {seg['start']:7.2f}s - {seg['end']:7.2f}s  ({dur:.2f}s)  {seg['name']}")

        print(f"\nDistinct tracks: {sorted(track_ids)}")
        print(f"Total segments:  {len(segments)}")
