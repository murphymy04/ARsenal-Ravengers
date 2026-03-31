"""Speaking segment logger for speaker-transcript alignment.

Records who is speaking and when, producing a timeline of segments:
    [{"person_id": 1, "name": "Alice", "start": 1710000000.1, "end": 1710000003.4}, ...]

Later, Whisper transcript segments (which carry their own start/end times) can be
matched to speakers by finding the speaking segment that overlaps each text segment.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Segments shorter than this are discarded as noise (seconds)
_MIN_SEGMENT_DURATION = 0.4


@dataclass
class SpeakingSegment:
    track_id: int
    person_id: Optional[int]
    name: str
    start: float
    end: Optional[float] = None  # None while still speaking

    def duration(self) -> float:
        if self.end is None:
            return time.time() - self.start
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "person_id": self.person_id,
            "name": self.name,
            "start": round(self.start, 3),
            "end": round(self.end or time.time(), 3),
        }


class SpeakingLog:
    """Tracks speaking state transitions and accumulates completed segments.

    Call update() once per video frame for each visible face.
    Call save() or close() when the session ends.
    """

    def __init__(self):
        # track_id -> open (in-progress) segment
        self._open: dict[int, SpeakingSegment] = {}
        # completed segments, ordered by start time
        self._segments: list[SpeakingSegment] = []

    def update(
        self,
        track_id: int,
        person_id: Optional[int],
        name: str,
        is_speaking: bool,
        timestamp: Optional[float] = None,
    ):
        """Call once per frame per face."""
        ts = timestamp or time.time()

        if is_speaking:
            if track_id not in self._open:
                # Speaking just started
                self._open[track_id] = SpeakingSegment(
                    track_id=track_id,
                    person_id=person_id,
                    name=name,
                    start=ts,
                )
            else:
                # Still speaking — keep name/id current (may have just been identified)
                seg = self._open[track_id]
                seg.person_id = person_id
                seg.name = name
        else:
            if track_id in self._open:
                # Speaking just stopped — close the segment
                seg = self._open.pop(track_id)
                seg.end = ts
                if seg.duration() >= _MIN_SEGMENT_DURATION:
                    self._segments.append(seg)

    def close(self, timestamp: Optional[float] = None):
        """Finalize any open segments (call when session ends)."""
        ts = timestamp or time.time()
        for seg in self._open.values():
            seg.end = ts
            if seg.duration() >= _MIN_SEGMENT_DURATION:
                self._segments.append(seg)
        self._open.clear()

    def get_segments(self) -> list[dict]:
        """Return all completed segments as dicts, sorted by start time."""
        return sorted(
            [s.to_dict() for s in self._segments],
            key=lambda s: s["start"],
        )

    def save(self, path: Path):
        """Write the speaking timeline to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.close()
        segments = self.get_segments()
        with open(path, "w") as f:
            json.dump(segments, f, indent=2)
        print(f"\nSpeaking log saved → {path}  ({len(segments)} segments)")

    def assign_transcript(self, transcript_segments: list[dict]) -> list[dict]:
        """Assign a speaker name to each Whisper transcript segment.

        Matches each transcript segment to the speaking log entry with the
        most overlap.  Unmatched segments get speaker='unknown'.

        Args:
            transcript_segments: list of dicts with 'start', 'end', 'text' keys
                                  (Whisper output format).

        Returns:
            Same list with a 'speaker' key added to each segment.
        """
        completed = self.get_segments()
        result = []
        for seg in transcript_segments:
            t_start, t_end = seg["start"], seg["end"]
            best_name, best_overlap = "unknown", 0.0
            for sp in completed:
                overlap = min(t_end, sp["end"]) - max(t_start, sp["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_name = sp["name"]
            result.append({**seg, "speaker": best_name})
        return result
