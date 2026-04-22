"""Diarization pipeline: face detection + identity + speaking detection.

Processes frames and audio, producing a speaking log with per-segment
identity labels. Accepts a pluggable IdentityModule — defaults to
NullIdentity (no face recognition, track-ID-only labels).

Exposes process_frame() for streaming use cases and run() for batch
processing.
"""

import sys
from pathlib import Path

import numpy as np

_AR_ROOT = Path(__file__).resolve().parent.parent
if str(_AR_ROOT) not in sys.path:
    sys.path.insert(0, str(_AR_ROOT))

from models import IdentityModule
from pipeline.identity import NullIdentity
from pipeline.retrieval import RetrievalDispatcher
from pipeline.segments import merge_close_segments
from processing.face_detector import FaceDetector
from processing.face_tracker import FaceTracker
from processing.speaking_detector import SpeakingDetector
from processing.vad_speaker import VadSpeaker
from storage.speaking_log import SpeakingLog

from config import SPEAKING_BACKEND


def _create_speaker(fps: float, static_boundary: float | None = None):
    if SPEAKING_BACKEND == "vad_rms":
        return VadSpeaker(fps=fps, static_boundary=static_boundary)
    return SpeakingDetector(fps=fps)


class DiarizationPipeline:
    def __init__(
        self,
        identity: IdentityModule | None = None,
        track_event_queue=None,
        static_boundary: float | None = None,
    ):
        self._identity = identity or NullIdentity()
        self._static_boundary = static_boundary
        self._detector: FaceDetector | None = None
        self._tracker: FaceTracker | None = None
        self._speaker = None
        self._log: SpeakingLog | None = None
        self._last_faces: list = []
        self._last_smoothed: list = []
        self._last_track_ids: list = []
        self._dispatcher = (
            RetrievalDispatcher(track_event_queue.put_nowait)
            if track_event_queue
            else None
        )
        self._owns_vision = True

    def open(self, fps: float, owns_vision: bool = True):
        self._owns_vision = owns_vision
        if owns_vision:
            self._detector = FaceDetector()
            self._tracker = FaceTracker()
        self._speaker = _create_speaker(fps, self._static_boundary)
        self._log = SpeakingLog()
        self._last_faces = []
        self._last_smoothed = []
        self._last_track_ids = []

    def process_frame(
        self,
        frame: np.ndarray,
        audio_chunk: np.ndarray,
        frame_idx: int,
        timestamp: float,
        is_vision_frame: bool = True,
        vision_result=None,
    ) -> tuple[int, int]:
        self._speaker.drip_audio(audio_chunk)

        if vision_result is not None:
            faces = vision_result.faces
            smoothed = vision_result.smoothed_matches
            track_ids = vision_result.track_ids
            for face, tid in zip(faces, track_ids, strict=False):
                self._speaker.add_crop(tid, face.crop)
            self._last_faces = faces
            self._last_smoothed = smoothed
            self._last_track_ids = track_ids
        elif is_vision_frame and self._owns_vision:
            faces, smoothed, track_ids = self._run_vision_in_band(
                frame, frame_idx, timestamp
            )
        else:
            faces = self._last_faces
            smoothed = self._last_smoothed
            track_ids = self._last_track_ids

        self._speaker.run_inference(frame_idx, active_track_ids=set(track_ids))

        speaking_count = 0
        for _face, match, tid in zip(faces, smoothed, track_ids, strict=False):
            is_speaking = self._speaker.get_speaking(tid)
            if is_speaking:
                speaking_count += 1
            self._log.update(
                track_id=tid,
                person_id=match.person_id if match.is_known else None,
                name=match.name if match.is_known else f"track_{tid}",
                is_speaking=is_speaking,
                timestamp=timestamp,
            )

        return len(faces), speaking_count

    def _run_vision_in_band(
        self, frame: np.ndarray, frame_idx: int, timestamp: float
    ) -> tuple[list, list, list[int]]:
        faces = self._detector.detect(frame, timestamp=timestamp)
        raw_matches = [self._identity.identify(face, frame_idx) for face in faces]
        smoothed, track_ids, new_track_ids = self._tracker.update(
            faces, raw_matches, frame_idx
        )
        if self._dispatcher:
            self._dispatcher.dispatch(smoothed, track_ids, new_track_ids, timestamp)
        for face, tid in zip(faces, track_ids, strict=False):
            self._speaker.add_crop(tid, face.crop)
        self._last_faces = faces
        self._last_smoothed = smoothed
        self._last_track_ids = track_ids
        return faces, smoothed, track_ids

    def take_segments(self, timestamp: float) -> list[dict]:
        self._log.close(timestamp=timestamp)
        segments = merge_close_segments(self._log.get_segments())
        self._log = SpeakingLog()
        return segments

    def close(self):
        self._speaker.close()
        if self._owns_vision and self._detector is not None:
            self._detector.close()

    def run(
        self, frames: list[np.ndarray], fps: float, audio: np.ndarray
    ) -> list[dict]:
        self.open(fps)
        samples_per_frame = len(audio) / len(frames) if frames else 0
        try:
            for frame_idx, frame in enumerate(frames):
                timestamp = frame_idx / fps
                audio_start = int(frame_idx * samples_per_frame)
                audio_end = int((frame_idx + 1) * samples_per_frame)
                chunk = audio[audio_start:audio_end]
                self.process_frame(frame, chunk, frame_idx, timestamp)
        finally:
            self.close()
        return self.take_segments(len(frames) / fps)

    @staticmethod
    def test(segments: list[dict]):
        print(f"\n{'=' * 60}")
        print(f"Speaking log: {len(segments)} segments")
        print(f"{'=' * 60}")

        track_ids = set()
        for seg in segments:
            track_ids.add(seg["track_id"])
            dur = seg["end"] - seg["start"]
            print(
                f"  track={seg['track_id']:2d}  "
                f"{seg['start']:7.2f}s - {seg['end']:7.2f}s  "
                f"({dur:.2f}s)  {seg['name']}"
            )

        print(f"\nDistinct tracks: {sorted(track_ids)}")
        print(f"Total segments:  {len(segments)}")
