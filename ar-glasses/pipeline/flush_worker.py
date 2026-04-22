"""Background worker that runs window-flush processing off the capture thread.

Groq Whisper transcription and recording-buffer ingestion used to run inline
in LivePipelineDriver.flush_window, blocking the consumer loop for 1-3s
every LIVE_BUFFER_SECONDS. The worker consumes (segments, audio, window_start,
window_end) tuples from a queue and produces (combined, diarization_segments)
results on a result queue the main thread drains at its own cadence.
"""

import queue
import threading
from collections.abc import Callable

import numpy as np
from pipeline.driver import combine_segments
from pipeline.recording_buffer import (
    ChunkData,
    RecordingBuffer,
    detect_long_turn,
)
from pipeline.transcription import TranscriptionPipeline


class FlushWorker:
    def __init__(
        self,
        transcription: TranscriptionPipeline,
        recording_buffer: RecordingBuffer,
        pcm_to_wav: Callable[[np.ndarray], bytes],
        sanitize_and_flush: Callable[[list[ChunkData]], None],
    ):
        self._transcription = transcription
        self._recording_buffer = recording_buffer
        self._pcm_to_wav = pcm_to_wav
        self._sanitize_and_flush = sanitize_and_flush
        self._work_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def submit(
        self,
        diarization_segments: list[dict],
        audio: np.ndarray,
        window_start: float,
        window_end: float,
    ):
        self._work_queue.put((diarization_segments, audio, window_start, window_end))

    def drain_results(self) -> list[tuple[list[dict], list[dict]]]:
        results: list[tuple[list[dict], list[dict]]] = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                return results

    def stop(self, timeout: float = 180):
        self._work_queue.put(None)
        self._thread.join(timeout=timeout)

    def _run(self):
        while True:
            item = self._work_queue.get()
            if item is None:
                return
            self._process_window(*item)

    def _process_window(
        self,
        diarization_segments: list[dict],
        audio: np.ndarray,
        window_start: float,
        window_end: float,
    ):
        if len(audio) == 0:
            print(f"\n[{window_start:.1f}s - {window_end:.1f}s] No audio captured")
            self._result_queue.put(([], diarization_segments))
            return

        wav_bytes = self._pcm_to_wav(audio)
        transcript_segments = self._transcription.run(wav_bytes)

        for seg in transcript_segments:
            seg.start_time += window_start
            seg.end_time += window_start

        combined = combine_segments(diarization_segments, transcript_segments)

        long_turn = detect_long_turn(diarization_segments, window_start, window_end)
        chunk = ChunkData(
            audio=audio,
            diarization_segments=diarization_segments,
            transcript_segments=transcript_segments,
            combined=combined,
            window_start=window_start,
            window_end=window_end,
        )
        flushable = self._recording_buffer.ingest(chunk, long_turn)
        if flushable is not None:
            self._sanitize_and_flush(flushable)

        self._log_window(window_start, window_end, diarization_segments, combined)
        self._result_queue.put((combined, diarization_segments))

    def _log_window(
        self,
        window_start: float,
        window_end: float,
        diarization_segments: list[dict],
        combined: list[dict],
    ):
        print(f"\n[{window_start:.1f}s - {window_end:.1f}s]")
        print(f"  ASD segments ({len(diarization_segments)}):")
        for seg in diarization_segments:
            print(
                f"    [{seg['start']:7.2f} - {seg['end']:7.2f}] "
                f"{seg['name']} (track {seg['track_id']})"
            )
        print(f"  Combined ({len(combined)}):")
        for seg in combined:
            print(
                f"    [{seg['start']:7.2f} - {seg['end']:7.2f}] "
                f"{seg['speaker']}: {seg['text']}"
            )

        flag = "ON " if self._recording_buffer.flag else "OFF"
        print(
            f"===[ recording={flag} | buffered={len(self._recording_buffer.chunks)} "
            f"chunks | quiet={self._recording_buffer.quiet_chunks} ]==="
        )
