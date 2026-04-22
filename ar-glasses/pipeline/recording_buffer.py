"""Recording flag heuristic for the live pipeline.

Replaces the LLM-based conversation-end detection. The recording flag
flips on when a non-wearer holds a sustained turn within a chunk and
flips off after a configurable number of consecutive quiet chunks. While
on, audio + diarization + transcript chunks are buffered for deferred
sanitization and flush.

Redundant Zep episodes are acceptable — they are synthesized at retrieval
time — so this heuristic favours recording too much over too little.
"""

import io
import tempfile
import wave
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from groq import Groq
from pipeline.config import (
    DIARIZATION_QUIET_CHUNKS_TO_FLUSH,
    DIARIZATION_TURN_DOMINANCE,
    DIARIZATION_TURN_LENGTH,
)

from config import SAMPLE_RATE


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


@dataclass
class SanitizedConversation:
    spoke_with: str
    person_id: int | None
    transcript: str
    window_start: float
    window_end: float


_groq = Groq()


def pcm_to_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _transcribe(wav_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(wav_bytes)
    try:
        with open(tmp_path, "rb") as f:
            response = _groq.audio.transcriptions.create(
                file=(tmp_path.name, f),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
        return " ".join(seg["text"].strip() for seg in response.segments)
    finally:
        tmp_path.unlink(missing_ok=True)


def sanitize(buffer: list[ChunkData]) -> SanitizedConversation:
    all_audio = np.concatenate([chunk.audio for chunk in buffer])
    wav_bytes = pcm_to_wav(all_audio)
    transcript = _transcribe(wav_bytes)

    counts: Counter[tuple[int | None, str]] = Counter()
    for chunk in buffer:
        for seg in chunk.diarization_segments:
            counts[(seg["person_id"], seg["name"])] += 1

    if counts:
        (person_id, name), _ = counts.most_common(1)[0]
    else:
        person_id, name = None, "Unknown"

    return SanitizedConversation(
        spoke_with=name,
        person_id=person_id,
        transcript=transcript,
        window_start=buffer[0].window_start,
        window_end=buffer[-1].window_end,
    )


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
