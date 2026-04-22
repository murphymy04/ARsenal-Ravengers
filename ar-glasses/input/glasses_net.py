r"""ADB/scrcpy-based video and audio streaming from USB-connected glasses.

Replaces the WiFi-based DiscoveryService, VideoReceiver (UDP port 5000), and
AudioReceiver (TCP port 5003) with a single ScrcpyStream that drives scrcpy
and one FFmpeg subprocess over USB. No WiFi required.

scrcpy records its mkv output to a POSIX FIFO. A single FFmpeg reads that FIFO
directly and demuxes both streams in one pass -- video goes to one os.pipe()
as raw BGR, audio goes to another as PCM s16le. Reader threads drain each pipe
into bounded deques.

AdbWatcher polls `adb devices` and runs `adb forward tcp:HUD_PORT tcp:HUD_PORT`
on each new device so the glasses Unity app can reach HudBroadcastServer at
ws://localhost:HUD_PORT without any WiFi.
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from input.config import (
    ANDROID_CAMERA_FPS,
    GLASSES_VIDEO_HEIGHT,
    GLASSES_VIDEO_WIDTH,
    SAMPLE_RATE,
)
from pipeline.config import HUD_BROADCAST_PORT

AUDIO_CHUNK_SAMPLES = 1600  # 100 ms at 16 kHz
_ADB_POLL_INTERVAL = 2.0


@dataclass
class FrameData:
    seq: int
    timestamp: int  # ms from stream start
    width: int
    height: int
    data: np.ndarray
    received_at: float


@dataclass
class AudioChunk:
    seq: int
    timestamp_start: int  # ms from stream start
    timestamp_end: int
    sample_rate: int
    num_samples: int
    data: np.ndarray
    received_at: float


@dataclass
class GlassesConnection:
    serial: str


def _adb(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", *args], capture_output=True, text=True)


def _make_pipe_path() -> str:
    uid = uuid.uuid4().hex[:8]
    tmpdir = tempfile.mkdtemp(prefix="ar_glasses_")
    return os.path.join(tmpdir, f"cam_{uid}.mkv")


# ---------------------------------------------------------------------------
# ADB device watcher
# ---------------------------------------------------------------------------


class AdbWatcher:
    def __init__(
        self,
        on_device_connected: Callable[[str], None],
        on_device_disconnected: Callable[[str], None],
        hud_port: int = HUD_BROADCAST_PORT,
    ):
        self._on_connected = on_device_connected
        self._on_disconnected = on_device_disconnected
        self._hud_port = hud_port
        self._known: set[str] = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print("[ADB] Watching for USB devices")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def _poll_loop(self):
        while self._running:
            current = self._connected_serials()
            for serial in current - self._known:
                self._forward_hud_port(serial)
                self._known.add(serial)
                self._on_connected(serial)
            for serial in self._known - current:
                self._known.discard(serial)
                self._on_disconnected(serial)
            time.sleep(_ADB_POLL_INTERVAL)

    def _connected_serials(self) -> set[str]:
        result = _adb("devices")
        serials = set()
        for line in result.stdout.splitlines()[1:]:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "device":
                serials.add(parts[0])
        return serials

    def _forward_hud_port(self, serial: str):
        result = _adb(
            "-s",
            serial,
            "forward",
            f"tcp:{self._hud_port}",
            f"tcp:{self._hud_port}",
        )
        if result.returncode == 0:
            print(f"[ADB] HUD port {self._hud_port} forwarded for {serial}")
        else:
            print(f"[ADB] Port forward failed for {serial}: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# Scrcpy stream
# ---------------------------------------------------------------------------


class ScrcpyStream:
    """Camera video and microphone audio from a USB-connected Android device.

    scrcpy writes mkv to a POSIX FIFO; a single FFmpeg reads the FIFO and
    demuxes both tracks, streaming raw BGR video and PCM s16le audio to two
    os.pipe() file descriptors. Reader threads drain each into bounded deques.

    Exposes the same interface as the old VideoReceiver + AudioReceiver so
    PairingLoop requires no changes.
    """

    _MS_PER_FRAME = 1000 // ANDROID_CAMERA_FPS
    _MS_PER_CHUNK = AUDIO_CHUNK_SAMPLES * 1000 // SAMPLE_RATE
    _FRAME_BYTES = GLASSES_VIDEO_WIDTH * GLASSES_VIDEO_HEIGHT * 3  # bgr24
    _CHUNK_BYTES = AUDIO_CHUNK_SAMPLES * 2  # int16 LE

    def __init__(self, max_frames: int = 120, max_chunks: int = 1000):
        self._frames: deque[FrameData] = deque(maxlen=max_frames)
        self._frame_lock = threading.Lock()
        self._chunks: deque[AudioChunk] = deque(maxlen=max_chunks)
        self._chunks_lock = threading.Lock()
        self._running = False
        self._procs: list[subprocess.Popen] = []
        self._threads: list[threading.Thread] = []
        self._pipe_path: Optional[str] = None
        self.frames_received = 0
        self.chunks_received = 0
        self.start_time: Optional[float] = None

    def connect(self, serial: str):
        if self._running:
            self.disconnect()
        self._running = True
        self.start_time = time.time()
        self.frames_received = 0
        self.chunks_received = 0
        with self._frame_lock:
            self._frames.clear()
        with self._chunks_lock:
            self._chunks.clear()
        threading.Thread(target=self._setup, args=(serial,), daemon=True).start()

    def disconnect(self):
        self._running = False
        for proc in self._procs:
            try:
                proc.terminate()
            except Exception:
                pass
        self._procs = []
        for t in self._threads:
            t.join(timeout=2)
        self._threads = []
        if self._pipe_path:
            try:
                os.unlink(self._pipe_path)
                os.rmdir(os.path.dirname(self._pipe_path))
            except Exception:
                pass
            self._pipe_path = None
        print("[Scrcpy] Disconnected")

    def get_frame_after(self, seq: int) -> Optional[FrameData]:
        with self._frame_lock:
            for frame in self._frames:
                if frame.seq > seq:
                    return frame
        return None

    def get_audio_range(self, ts_start: int, ts_end: int) -> list[AudioChunk]:
        with self._chunks_lock:
            return [
                c
                for c in self._chunks
                if c.timestamp_end >= ts_start and c.timestamp_start <= ts_end
            ]

    def get_latest_audio_timestamp(self) -> Optional[int]:
        with self._chunks_lock:
            return self._chunks[-1].timestamp_end if self._chunks else None

    def get_next_chunk(self, after_seq: int) -> Optional[AudioChunk]:
        with self._chunks_lock:
            for chunk in self._chunks:
                if chunk.seq > after_seq:
                    return chunk
        return None

    def get_stats(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "frames_received": self.frames_received,
            "frames_buffered": len(self._frames),
            "frames_dropped": 0,
            "incomplete_dropped": 0,
            "fps": self.frames_received / elapsed if elapsed > 0 else 0,
            "elapsed": elapsed,
            "chunks_received": self.chunks_received,
            "chunks_buffered": len(self._chunks),
            "chunks_per_sec": self.chunks_received / elapsed if elapsed > 0 else 0,
            "connected": self._running,
        }

    # --- setup --------------------------------------------------------------

    def _setup(self, serial: str):
        missing = [t for t in ("scrcpy", "ffmpeg") if shutil.which(t) is None]
        if missing:
            print(f"[Scrcpy] missing tools not on PATH: {missing}")
            self.disconnect()
            return

        pipe_path = _make_pipe_path()
        os.mkfifo(pipe_path)
        self._pipe_path = pipe_path

        video_r, video_w = os.pipe()
        audio_r, audio_w = os.pipe()

        scrcpy_proc = self._spawn_scrcpy(serial, pipe_path)
        ffmpeg_proc = self._spawn_ffmpeg(pipe_path, video_w, audio_w)
        os.close(video_w)
        os.close(audio_w)
        self._procs = [scrcpy_proc, ffmpeg_proc]

        video_stream = os.fdopen(video_r, "rb", 0)
        audio_stream = os.fdopen(audio_r, "rb", 0)

        self._threads = [
            threading.Thread(
                target=self._read_video, args=(video_stream,), daemon=True
            ),
            threading.Thread(
                target=self._read_audio, args=(audio_stream,), daemon=True
            ),
        ]
        for t in self._threads:
            t.start()
        print(f"[Scrcpy] Streaming from {serial}")

    def _spawn_scrcpy(self, serial: str, pipe_path: str) -> subprocess.Popen:
        cmd = [
            "scrcpy",
            "-s",
            serial,
            "--force-adb-forward",
            "--video-source=camera",
            "--camera-facing=back",
            f"--camera-fps={ANDROID_CAMERA_FPS}",
            f"--max-size={GLASSES_VIDEO_WIDTH}",
            "--video-codec=h264",
            "--video-encoder=c2.qti.avc.encoder",
            "--audio-source=mic-camcorder",
            "--no-playback",
            "--no-window",
            f"--record={pipe_path}",
            "--record-format=mkv",
        ]
        print(f"[Scrcpy] launching: {' '.join(cmd)}")
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL)

    def _spawn_ffmpeg(
        self, input_path: str, video_fd: int, audio_fd: int
    ) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer+discardcorrupt",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-i",
            input_path,
            "-map",
            "0:v",
            "-vf",
            f"scale={GLASSES_VIDEO_WIDTH}:{GLASSES_VIDEO_HEIGHT}",
            "-r",
            str(ANDROID_CAMERA_FPS),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            f"pipe:{video_fd}",
            "-map",
            "0:a",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            f"pipe:{audio_fd}",
        ]
        return subprocess.Popen(cmd, pass_fds=(video_fd, audio_fd))

    # --- readers ------------------------------------------------------------

    def _recv_exact(self, src, n: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < n and self._running:
            chunk = src.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None

    def _read_video(self, stream):
        seq = 0
        while self._running:
            raw = self._recv_exact(stream, self._FRAME_BYTES)
            if raw is None:
                break
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                (GLASSES_VIDEO_HEIGHT, GLASSES_VIDEO_WIDTH, 3)
            )
            frame = FrameData(
                seq=seq,
                timestamp=seq * self._MS_PER_FRAME,
                width=GLASSES_VIDEO_WIDTH,
                height=GLASSES_VIDEO_HEIGHT,
                data=arr.copy(),
                received_at=time.time(),
            )
            with self._frame_lock:
                self._frames.append(frame)
            if seq == 0:
                print("[Scrcpy] first video frame received")
            seq += 1
            self.frames_received += 1
        print("[Scrcpy] Video stream ended")

    def _read_audio(self, stream):
        seq = 0
        while self._running:
            raw = self._recv_exact(stream, self._CHUNK_BYTES)
            if raw is None:
                break
            audio = np.frombuffer(raw, dtype=np.int16).copy()
            ts_start = seq * self._MS_PER_CHUNK
            chunk = AudioChunk(
                seq=seq,
                timestamp_start=ts_start,
                timestamp_end=ts_start + self._MS_PER_CHUNK,
                sample_rate=SAMPLE_RATE,
                num_samples=AUDIO_CHUNK_SAMPLES,
                data=audio,
                received_at=time.time(),
            )
            with self._chunks_lock:
                self._chunks.append(chunk)
            if seq == 0:
                print("[Scrcpy] first audio chunk received")
            seq += 1
            self.chunks_received += 1
        print("[Scrcpy] Audio stream ended")
