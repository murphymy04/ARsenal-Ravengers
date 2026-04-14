"""Adapter between the glasses stream protocol and the live pipeline.

The glasses publish video (UDP, lossy) and audio (TCP, lossless) with their own
clock embedded in packet headers. The PairingLoop consumes both receivers and
emits (frame, audio_slice, ts) triples using glasses timestamps as the PTS —
each frame carries the audio slice covering [prev_frame_ts, this_frame_ts].

A laptop-side stopwatch paces emission so downstream sees real-time playback
regardless of network burstiness. On audio reconnect or video silence, both
receivers and the pairing loop reset together so stale packets never leak
across a stream break.

GlassesCamera and GlassesMic expose the same surface pipeline/live.py already
uses for Camera and Microphone.
"""

import queue
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from config import (
    GLASSES_PAIR_QUEUE_MAX,
    GLASSES_PREBUFFER_SECONDS,
    GLASSES_SPIN_INTERVAL_SEC,
)
from input.glasses_net import (
    AudioChunk,
    AudioReceiver,
    DiscoveryService,
    VideoReceiver,
)

REANCHOR_THRESHOLD_SEC = 1.0
STATS_INTERVAL_SEC = 2.0


def slice_audio_by_timestamp(
    chunks: list[AudioChunk], ts_start: int, ts_end: int
) -> np.ndarray:
    if not chunks or ts_end <= ts_start:
        return np.zeros(0, dtype=np.float32)

    parts: list[np.ndarray] = []
    for chunk in chunks:
        span = chunk.timestamp_end - chunk.timestamp_start
        if span <= 0 or chunk.num_samples == 0:
            continue

        overlap_start = max(ts_start, chunk.timestamp_start)
        overlap_end = min(ts_end, chunk.timestamp_end)
        if overlap_end <= overlap_start:
            continue

        start_frac = (overlap_start - chunk.timestamp_start) / span
        end_frac = (overlap_end - chunk.timestamp_start) / span
        start_sample = int(start_frac * chunk.num_samples)
        end_sample = int(end_frac * chunk.num_samples)
        if end_sample <= start_sample:
            continue

        parts.append(chunk.data[start_sample:end_sample])

    if not parts:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(parts).astype(np.float32) / 32768.0


class PairingLoop:
    def __init__(
        self,
        video_rx: VideoReceiver,
        audio_rx: AudioReceiver,
        max_pairs: int = GLASSES_PAIR_QUEUE_MAX,
    ):
        self._video_rx = video_rx
        self._audio_rx = audio_rx
        self.pairs: queue.Queue = queue.Queue(maxsize=max_pairs)

        self._running = False
        self._reset_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_paired_seq = -1
        self._last_paired_ts: Optional[int] = None
        self._first_ts: Optional[int] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def reset(self):
        self._reset_event.set()
        self._drain_pairs()

    def _run(self):
        while self._running:
            if not self._seed():
                return
            self._stream()

    def _seed(self) -> bool:
        self._reset_event.clear()
        self._last_paired_seq = -1
        self._last_paired_ts = None
        self._first_ts = None

        first_frame = self._wait_for_first_frame()
        if first_frame is None:
            return False

        self._first_ts = first_frame.timestamp
        self._last_paired_ts = first_frame.timestamp
        self._last_paired_seq = first_frame.seq
        print(f"[Pairing] seeded first_ts={first_frame.timestamp}")

        prebuffer_target = first_frame.timestamp + int(GLASSES_PREBUFFER_SECONDS * 1000)
        if not self._wait_for_audio(prebuffer_target):
            return self._running

        print(f"[Pairing] prebuffer filled ({GLASSES_PREBUFFER_SECONDS:.1f}s)")
        return True

    def _stream(self):
        stopwatch_start = time.perf_counter()
        emit_first_ts: Optional[float] = None
        last_stats_at = time.monotonic()

        while self._running and not self._reset_event.is_set():
            now = time.monotonic()
            if now - last_stats_at >= STATS_INTERVAL_SEC:
                last_stats_at = now
                self._log_stats()

            frame = self._video_rx.get_frame_after(self._last_paired_seq)
            if frame is None:
                time.sleep(GLASSES_SPIN_INTERVAL_SEC)
                continue

            if not self._wait_for_audio(frame.timestamp):
                return

            chunks = self._audio_rx.get_audio_range(
                self._last_paired_ts, frame.timestamp
            )
            audio = slice_audio_by_timestamp(
                chunks, self._last_paired_ts, frame.timestamp
            )
            bgr = cv2.cvtColor(frame.data, cv2.COLOR_GRAY2BGR)
            ts_seconds = (frame.timestamp - self._first_ts) / 1000.0

            if emit_first_ts is None:
                emit_first_ts = ts_seconds

            deadline = stopwatch_start + (ts_seconds - emit_first_ts)
            wall_now = time.perf_counter()
            if wall_now < deadline:
                time.sleep(deadline - wall_now)
            elif wall_now - deadline > REANCHOR_THRESHOLD_SEC:
                stopwatch_start = time.perf_counter()
                emit_first_ts = ts_seconds
                print("[Pairing] emission re-anchored (fell >1s behind)")

            self._put_pair((bgr, audio, ts_seconds))
            self._last_paired_seq = frame.seq
            self._last_paired_ts = frame.timestamp

    def _wait_for_first_frame(self):
        while self._running and not self._reset_event.is_set():
            frame = self._video_rx.get_frame_after(-1)
            if frame is not None:
                return frame
            time.sleep(GLASSES_SPIN_INTERVAL_SEC)
        return None

    def _wait_for_audio(self, target_ts: int) -> bool:
        while self._running and not self._reset_event.is_set():
            latest = self._audio_rx.get_latest_audio_timestamp()
            if latest is not None and latest >= target_ts:
                return True
            time.sleep(GLASSES_SPIN_INTERVAL_SEC)
        return False

    def _put_pair(self, pair):
        try:
            self.pairs.put_nowait(pair)
            return
        except queue.Full:
            pass
        try:
            self.pairs.get_nowait()
        except queue.Empty:
            pass
        self.pairs.put_nowait(pair)

    def _drain_pairs(self):
        while True:
            try:
                self.pairs.get_nowait()
            except queue.Empty:
                return

    def _log_stats(self):
        latest_audio_ts = self._audio_rx.get_latest_audio_timestamp()
        print(
            f"[Pairing] stats: "
            f"v_recv={self._video_rx.frames_received} "
            f"v_drop={self._video_rx.frames_dropped} | "
            f"a_recv={self._audio_rx.chunks_received} "
            f"a_ts={latest_audio_ts} | "
            f"pair_q={self.pairs.qsize()}"
        )


class GlassesMic:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._pending_advance: list[np.ndarray] = []
        self._window_buffer: list[np.ndarray] = []

    def deliver(self, audio: np.ndarray):
        with self._lock:
            self._pending_advance.append(audio)
            self._window_buffer.append(audio)

    def advance_frame(self) -> np.ndarray:
        with self._lock:
            if not self._pending_advance:
                return np.zeros(0, dtype=np.float32)
            buf = np.concatenate(self._pending_advance)
            self._pending_advance.clear()
            return buf

    def get_buffer_and_clear(self) -> np.ndarray:
        with self._lock:
            if not self._window_buffer:
                return np.zeros(0, dtype=np.float32)
            buf = np.concatenate(self._window_buffer)
            self._window_buffer.clear()
            return buf

    def open(self):
        pass

    def close(self):
        pass


class GlassesCamera:
    def __init__(self, loop: PairingLoop, mic: GlassesMic):
        self._loop = loop
        self._mic = mic
        self._running = True
        self._last_ts_seconds = 0.0

    def frames(self):
        while self._running:
            try:
                bgr, audio, ts_seconds = self._loop.pairs.get(timeout=1.0)
            except queue.Empty:
                if not self._running:
                    break
                continue
            self._mic.deliver(audio)
            self._last_ts_seconds = ts_seconds
            yield bgr

    @property
    def is_opened(self) -> bool:
        return self._running

    @property
    def last_timestamp_seconds(self) -> float:
        return self._last_ts_seconds

    def close(self):
        self._running = False


class GlassesServer:
    def __init__(self, sample_rate: int = 16000):
        self.discovery: Optional[DiscoveryService] = None
        self.audio_rx = AudioReceiver()
        self.video_rx = VideoReceiver()
        self.pairing = PairingLoop(self.video_rx, self.audio_rx)
        self.mic = GlassesMic(sample_rate)
        self.camera = GlassesCamera(self.pairing, self.mic)

        self.audio_rx.on_reconnect_callback = self._on_audio_reconnect
        self.video_rx.on_silence_callback = self._on_video_silence

    def start(self) -> tuple[GlassesCamera, GlassesMic, Callable[[int], float]]:
        self.discovery = DiscoveryService()
        self.discovery.start()
        self.audio_rx.start()
        self.video_rx.start()
        self.pairing.start()

        def clock_fn(frame_idx: int) -> float:
            return self.camera.last_timestamp_seconds

        return self.camera, self.mic, clock_fn

    def stop(self):
        self.camera.close()
        self.pairing.stop()
        self.video_rx.stop()
        self.audio_rx.stop()
        if self.discovery:
            self.discovery.stop()

    def _on_audio_reconnect(self):
        print("[GlassesServer] Audio reconnect — resetting video + pairing")
        self.video_rx.reset()
        self.pairing.reset()

    def _on_video_silence(self):
        print("[GlassesServer] Video silence — resetting pairing")
        self.pairing.reset()
