"""Adapter between the glasses stream protocol and the live pipeline.

The glasses publish video (UDP, lossy) and audio (TCP, lossless) with their own
clock embedded in packet headers. Wall-clock time on the laptop is not safe
because video and audio would drift independently. The PairingLoop consumes
both receivers and emits (frame, audio_slice, ts) triples using glasses
timestamps as the PTS — each frame carries the audio slice covering
[prev_frame_ts, this_frame_ts].

GlassesCamera and GlassesMic expose the same surface pipeline/live.py already
uses for Camera and Microphone, so no pipeline changes are required beyond a
new --glasses branch in __main__.
"""

import queue
import threading
import time
from collections import deque
from typing import Callable, Optional

import cv2
import numpy as np

from config import (
    GLASSES_DROP_LAG_SECONDS,
    GLASSES_MAX_STAGING_SECONDS,
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

    merged = np.concatenate(parts).astype(np.float32) / 32768.0
    return merged


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

        self._staging: deque = deque()
        self._staging_lock = threading.Lock()

        self._last_paired_seq = -1
        self._last_paired_ts: Optional[int] = None
        self._first_ts: Optional[int] = None
        self._logged_first = False

        self._running = False
        self._builder_thread: Optional[threading.Thread] = None
        self._emitter_thread: Optional[threading.Thread] = None

    def reset(self):
        with self._staging_lock:
            self._staging.clear()
        self._last_paired_seq = -1
        self._last_paired_ts = None
        self._first_ts = None
        while True:
            try:
                self.pairs.get_nowait()
            except queue.Empty:
                break

    def start(self):
        self._running = True
        self._builder_thread = threading.Thread(target=self._build_loop, daemon=True)
        self._emitter_thread = threading.Thread(target=self._emit_loop, daemon=True)
        self._builder_thread.start()
        self._emitter_thread.start()

    def stop(self):
        self._running = False
        if self._builder_thread:
            self._builder_thread.join(timeout=2)
        if self._emitter_thread:
            self._emitter_thread.join(timeout=2)

    def _build_loop(self):
        last_stats_print = time.time()

        while self._running:
            now = time.time()
            if now - last_stats_print >= 2.0:
                last_stats_print = now
                self._log_stats()

            frame = self._video_rx.get_frame_after(self._last_paired_seq)
            if frame is None:
                time.sleep(GLASSES_SPIN_INTERVAL_SEC)
                continue

            if self._last_paired_seq > 100 and frame.seq < 10:
                print(
                    f"[Pairing] seq reset: last={self._last_paired_seq}, new={frame.seq}"
                )
                self.reset()
                continue

            if self._last_paired_ts is None:
                self._first_ts = frame.timestamp
                self._last_paired_ts = frame.timestamp
                self._last_paired_seq = frame.seq
                print(f"[Pairing] seeded first_ts={frame.timestamp}")
                continue

            if not self._wait_for_audio(frame.timestamp):
                return

            last_ts = self._last_paired_ts
            first_ts = self._first_ts
            if last_ts is None or first_ts is None:
                continue

            chunks = self._audio_rx.get_audio_range(last_ts, frame.timestamp)
            audio = slice_audio_by_timestamp(chunks, last_ts, frame.timestamp)
            bgr = cv2.cvtColor(frame.data, cv2.COLOR_GRAY2BGR)
            ts_seconds = (frame.timestamp - first_ts) / 1000.0

            if not self._logged_first:
                self._logged_first = True
                print(
                    f"[Pairing] first pair built: ts={frame.timestamp} ms, "
                    f"audio_samples={audio.shape[0]}, ts_s={ts_seconds:.3f}"
                )

            with self._staging_lock:
                self._staging.append((bgr, audio, ts_seconds))
                while (
                    len(self._staging) > 1
                    and self._staging[-1][2] - self._staging[0][2]
                    > GLASSES_MAX_STAGING_SECONDS
                ):
                    self._staging.popleft()

            self._last_paired_seq = frame.seq
            self._last_paired_ts = frame.timestamp

    def _emit_loop(self):
        while self._running:
            if not self._staging_ready():
                time.sleep(GLASSES_SPIN_INTERVAL_SEC)
                continue

            emit_wall_start = time.perf_counter()
            emit_ts_start: Optional[float] = None
            carry_audio = np.zeros(0, dtype=np.float32)
            print(
                f"[Pairing] prebuffer filled ({GLASSES_PREBUFFER_SECONDS:.1f}s), "
                f"starting paced emission"
            )

            while self._running:
                with self._staging_lock:
                    if not self._staging:
                        pair = None
                    else:
                        pair = self._staging.popleft()

                if pair is None:
                    time.sleep(GLASSES_SPIN_INTERVAL_SEC)
                    continue

                _, _, ts_seconds = pair
                if emit_ts_start is None:
                    emit_ts_start = ts_seconds

                target_wall = emit_wall_start + (ts_seconds - emit_ts_start)
                lag = time.perf_counter() - target_wall

                if lag < 0:
                    time.sleep(-lag)
                elif lag > 1.0:
                    emit_wall_start = time.perf_counter()
                    emit_ts_start = ts_seconds
                    print("[Pairing] emission re-anchored (fell >1s behind)")
                elif lag > GLASSES_DROP_LAG_SECONDS:
                    _, dropped_audio, _ = pair
                    carry_audio = np.concatenate([carry_audio, dropped_audio])
                    continue

                if carry_audio.size:
                    bgr, audio, ts = pair
                    pair = (bgr, np.concatenate([carry_audio, audio]), ts)
                    carry_audio = np.zeros(0, dtype=np.float32)

                self._put_pair(pair)

    def _staging_ready(self) -> bool:
        with self._staging_lock:
            if len(self._staging) < 2:
                return False
            span = self._staging[-1][2] - self._staging[0][2]
            return span >= GLASSES_PREBUFFER_SECONDS

    def _wait_for_audio(self, target_ts: int) -> bool:
        while self._running:
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

    def _log_stats(self):
        v_stats = self._video_rx.get_stats()
        a_stats = self._audio_rx.get_stats()
        latest_audio_ts = self._audio_rx.get_latest_audio_timestamp()
        with self._staging_lock:
            staging_depth = len(self._staging)
            staging_span = (
                self._staging[-1][2] - self._staging[0][2]
                if len(self._staging) >= 2
                else 0.0
            )
        print(
            f"[Pairing] stats: "
            f"v_recv={v_stats['frames_received']} "
            f"v_buf={v_stats['frames_buffered']} "
            f"v_drop={v_stats['frames_dropped']} | "
            f"a_recv={a_stats['chunks_received']} "
            f"a_buf={a_stats['chunks_buffered']} "
            f"a_ts={latest_audio_ts} | "
            f"staging={staging_depth} ({staging_span:.2f}s) "
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

        self._audio_connected_once = False
        self.audio_rx.on_connect_callback = self._on_audio_connect

    def start(self) -> tuple[GlassesCamera, GlassesMic, Callable[[int], float]]:
        self.discovery = DiscoveryService(
            on_glasses_found=lambda ip: print(f"[GlassesServer] Glasses at {ip}")
        )
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

    def _on_audio_connect(self):
        if not self._audio_connected_once:
            self._audio_connected_once = True
            print("[GlassesServer] audio connected")
            return
        print("[GlassesServer] audio reconnected — resetting pairing loop")
        self.pairing.reset()
