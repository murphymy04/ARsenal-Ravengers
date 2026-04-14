"""Recording flag heuristic for the live pipeline.

Replaces the LLM-based conversation-end detection. The recording flag
flips on when a non-wearer holds a sustained turn within a chunk and
flips off after a configurable number of consecutive quiet chunks. While
on, audio + diarization + transcript chunks are buffered for deferred
sanitization and flush.

Redundant Zep episodes are acceptable — they are synthesized at retrieval
time — so this heuristic favours recording too much over too little.
"""

from dataclasses import dataclass, field

import numpy as np
from pipeline.config import (
    DIARIZATION_QUIET_CHUNKS_TO_FLUSH,
    DIARIZATION_TURN_DOMINANCE,
    DIARIZATION_TURN_LENGTH,
)


@dataclass
class ChunkData:
    audio: np.ndarray
    diarization_segments: list[dict]
    transcript_segments: list
    combined: list[dict]
    window_start: float
    window_end: float


@dataclass
class LongTurn:
    person_id: int | None
    name: str
    ratio: float


def detect_long_turn(
    diarization_segments: list[dict],
    window_start: float,
    window_end: float,
    turn_length: float = DIARIZATION_TURN_LENGTH,
    dominance: float = DIARIZATION_TURN_DOMINANCE,
    bin_size: float = 0.1,
) -> LongTurn | None:
    """Return the non-wearer who dominates any `turn_length` sub-window.

    Slides a window of `turn_length` seconds across the chunk in
    `bin_size` increments. If a single diarization identity fills
    `dominance` fraction of the window, that's a long turn. Diarization
    segments only contain non-wearer speakers (visible faces), so any
    match counts.
    """
    if not diarization_segments or window_end <= window_start:
        return None

    n_bins = max(1, round((window_end - window_start) / bin_size))
    win_bins = round(turn_length / bin_size)

    if win_bins == 0 or win_bins > n_bins:
        return None

    persons: list[tuple[int | None, str]] = []

    for seg in diarization_segments:
        key = (seg["person_id"], seg["name"])
        if key not in persons:
            persons.append(key)

    best: LongTurn | None = None

    for person_id, name in persons:
        mask = [0] * n_bins
        for seg in diarization_segments:
            if (seg["person_id"], seg["name"]) != (person_id, name):
                continue

            start_bin = max(0, int((seg["start"] - window_start) / bin_size))
            end_bin = min(n_bins, int((seg["end"] - window_start) / bin_size))

            for i in range(start_bin, end_bin):
                mask[i] = 1

        threshold = win_bins * dominance
        speaking_bins = sum(mask[:win_bins])

        for i in range(n_bins - win_bins + 1):
            if i > 0:
                speaking_bins += mask[i + win_bins - 1] - mask[i - 1]

            ratio = speaking_bins / win_bins

            if speaking_bins >= threshold and (best is None or ratio > best.ratio):
                best = LongTurn(person_id=person_id, name=name, ratio=ratio)
                break

    return best


def sanitize(buffer: list[ChunkData]) -> list[ChunkData]:
    """Stub: offline diarization pass over the buffered chunks.

    Will rerun diarization on the joined audio and rebuild the combined
    transcript with refined speaker assignments. For now, pass through
    unchanged so the rest of the pipeline can be exercised.
    """
    return buffer


@dataclass
class RecordingBuffer:
    quiet_chunks_to_flush: int = DIARIZATION_QUIET_CHUNKS_TO_FLUSH
    flag: bool = False
    chunks: list[ChunkData] = field(default_factory=list)
    quiet_chunks: int = 0
    previous_chunk: ChunkData | None = None

    def ingest(
        self, chunk: ChunkData, long_turn: LongTurn | None
    ) -> list[ChunkData] | None:
        """Return the buffered chunks to flush when the flag turns off."""
        if not self.flag:
            if long_turn is None:
                self.previous_chunk = chunk
                return None
            self.flag = True
            if self.previous_chunk is not None:
                self.chunks.append(self.previous_chunk)
            self.chunks.append(chunk)
            self.previous_chunk = None
            self.quiet_chunks = 0
            return None

        self.chunks.append(chunk)
        if long_turn is not None:
            self.quiet_chunks = 0
            return None

        self.quiet_chunks += 1
        if self.quiet_chunks < self.quiet_chunks_to_flush:
            return None

        flushed = self.chunks
        self.flag = False
        self.chunks = []
        self.quiet_chunks = 0
        self.previous_chunk = chunk
        return flushed

    def drain(self) -> list[ChunkData]:
        if not self.chunks:
            return []
        flushed = self.chunks
        self.flag = False
        self.chunks = []
        self.quiet_chunks = 0
        self.previous_chunk = None
        return flushed
