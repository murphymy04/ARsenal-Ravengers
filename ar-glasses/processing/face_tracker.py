"""Temporal identity smoothing via face tracking across frames (#9).

Associates detected faces across consecutive frames by bounding-box centre
proximity, maintains a rolling history of raw identity matches per track,
and returns the majority-vote identity for each face — eliminating the
per-frame flicker caused by occasional mismatches.
"""

from collections import Counter, deque

import numpy as np
from config import FACE_MAX_MOVE_PX, TEMPORAL_SMOOTHING_FRAMES
from models import DetectedFace, IdentityMatch


class FaceTracker:
    """Associates face detections across frames and smooths identity predictions.

    Each call to update() takes the raw per-frame match list and returns a
    smoothed list of the same length.  Internally it maintains lightweight
    tracks keyed by integer ID; tracks that disappear for more than a few
    frames are dropped automatically.
    """

    # Frames a track can go unseen before it is dropped
    _EXPIRY_FRAMES = 10

    def __init__(
        self,
        window: int = TEMPORAL_SMOOTHING_FRAMES,
        max_move_px: int = FACE_MAX_MOVE_PX,
    ):
        self._window = window
        self._max_move = max_move_px
        # track_id -> {'cx', 'cy', 'history': deque[IdentityMatch], 'last_frame': int}
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def update(
        self,
        faces: list[DetectedFace],
        raw_matches: list[IdentityMatch],
        frame_count: int,
    ) -> tuple[list[IdentityMatch], list[int]]:
        """Update tracks with this frame's detections and return smoothed matches.

        Args:
            faces: detected faces (in order).
            raw_matches: raw identity match for each face (same order).
            frame_count: current frame index (used for track expiry).

        Returns:
            Tuple of (smoothed IdentityMatch list, track_id list) — same order as faces.
        """
        assigned: set[int] = set()
        active_tracks: dict[int, dict] = {}
        smoothed: list[IdentityMatch] = []
        track_ids: list[int] = []

        for face, match in zip(faces, raw_matches, strict=False):
            cx, cy = face.bbox.center

            # Find the nearest unassigned existing track within max_move_px
            best_id, best_dist = None, float("inf")
            for tid, track in self._tracks.items():
                if tid in assigned:
                    continue
                dist = np.hypot(cx - track["cx"], cy - track["cy"])
                if dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is not None and best_dist <= self._max_move:
                # Update existing track — deque handles the window limit automatically
                track = self._tracks[best_id]
                track["history"].append(match)
                track["cx"], track["cy"] = cx, cy
                track["last_frame"] = frame_count
                assigned.add(best_id)
                active_tracks[best_id] = track
                track_ids.append(best_id)
            else:
                # Start a new track
                tid = self._next_id
                self._next_id += 1
                track = {
                    "cx": cx,
                    "cy": cy,
                    "history": deque([match], maxlen=self._window),
                    "last_frame": frame_count,
                }
                assigned.add(tid)
                active_tracks[tid] = track
                track_ids.append(tid)

            smoothed.append(_majority_vote(track["history"]))

        # Retain tracks that were active this frame OR disappeared very recently
        self._tracks = {
            tid: t
            for tid, t in {**self._tracks, **active_tracks}.items()
            if frame_count - t["last_frame"] <= self._EXPIRY_FRAMES
        }

        return smoothed, track_ids


def _majority_vote(history: list[IdentityMatch]) -> IdentityMatch:
    """Return the most-voted IdentityMatch from a history window.

    Ties are broken by recency (the most recent winning match is returned).
    """
    votes = Counter(m.person_id for m in history)
    winning_id = votes.most_common(1)[0][0]
    for m in reversed(history):
        if m.person_id == winning_id:
            return m
    return history[-1]
