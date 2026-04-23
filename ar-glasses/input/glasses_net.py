r"""ADB/scrcpy-based video and audio streaming from USB-connected glasses.

Replaces the WiFi-based DiscoveryService, VideoReceiver (UDP port 5000), and
AudioReceiver (TCP port 5003) with scrcpy + ffmpeg over USB. No WiFi required.

Platform handling for the scrcpy -> ffmpeg handoff:
  POSIX: scrcpy writes mkv to a FIFO; one ffmpeg reads the FIFO directly and
    demuxes both streams, emitting raw BGR video to one os.pipe() and PCM
    s16le audio to another (fds shared via Popen's pass_fds).
  Windows: Popen has no pass_fds and os.mkfifo doesn't exist. scrcpy writes
    mkv to a Windows named pipe; a relay thread tees the bytes into two
    ffmpeg subprocesses' stdin -- one decodes video, one decodes audio --
    and reader threads drain each ffmpeg's stdout.

AdbWatcher polls `adb devices` and runs `adb reverse tcp:HUD_DEVICE_PORT
tcp:HUD_HOST_PORT` on each new device so the glasses Unity app can reach
HudBroadcastServer at ws://localhost:HUD_DEVICE_PORT without any WiFi.
"""

import os
import shutil
import subprocess
import sys
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
from pipeline.config import HUD_BROADCAST_DEVICE_PORT, HUD_BROADCAST_PORT

AUDIO_CHUNK_SAMPLES = 1600  # 100 ms at 16 kHz
_ADB_POLL_INTERVAL = 2.0
_IS_WINDOWS = sys.platform == "win32"
_RELAY_CHUNK = 65536


@dataclass
class FrameData:
    seq: int
    timestamp: int  # ms from stream start
    width: int
    height: int
    data: np.ndarray


@dataclass
class AudioChunk:
    seq: int
    timestamp_start: int  # ms from stream start
    timestamp_end: int
    sample_rate: int
    num_samples: int
    data: np.ndarray


def _adb(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", *args], capture_output=True, text=True)


def _make_pipe_path() -> str:
    uid = uuid.uuid4().hex[:8]
    if _IS_WINDOWS:
        return rf"\\.\pipe\ar_glasses_{uid}"
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
        host_port: int = HUD_BROADCAST_PORT,
        device_port: int = HUD_BROADCAST_DEVICE_PORT,
    ):
        self._on_connected = on_device_connected
        self._on_disconnected = on_device_disconnected
        self._host_port = host_port
        self._device_port = device_port
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
                self._reverse_hud_port(serial)
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

    def _reverse_hud_port(self, serial: str):
        self._strip_stale_forward(serial)
        _adb("-s", serial, "reverse", "--remove", f"tcp:{self._device_port}")
        add = _adb(
            "-s",
            serial,
            "reverse",
            f"tcp:{self._device_port}",
            f"tcp:{self._host_port}",
        )
        if add.returncode != 0:
            print(f"[ADB] reverse failed for {serial}: {add.stderr.strip()}")
            return
        print(
            f"[ADB] reverse tcp:{self._device_port} -> tcp:{self._host_port} "
            f"for {serial}"
        )
        self._log_mappings(serial)

    def _strip_stale_forward(self, serial: str):
        for port in (self._host_port, self._device_port):
            _adb("-s", serial, "forward", "--remove", f"tcp:{port}")

    def _log_mappings(self, serial: str):
        reverse = _adb("-s", serial, "reverse", "--list").stdout.strip()
        forward = _adb("-s", serial, "forward", "--list").stdout.strip()
        print(f"[ADB] reverse --list ({serial}):\n{reverse or '  <empty>'}")
        print(f"[ADB] forward --list ({serial}):\n{forward or '  <empty>'}")


# ---------------------------------------------------------------------------
# Scrcpy stream
# ---------------------------------------------------------------------------


class ScrcpyStream:
    """Camera video and microphone audio from a USB-connected Android device.

    POSIX: scrcpy writes mkv to a FIFO; a single ffmpeg demuxes it into two
    os.pipe() fds shared via Popen pass_fds. Windows: scrcpy writes mkv to a
    named pipe; a relay thread tees bytes into two ffmpeg subprocesses' stdin
    (one video, one audio) because Popen lacks pass_fds on Windows.

    Either way, reader threads drain raw BGR video and PCM s16le audio into
    bounded deques.

    Exposes the same interface as the old VideoReceiver + AudioReceiver so
    PairingLoop requires no changes.
    """

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
        self.start_time: Optional[float] = None

    def connect(self, serial: str):
        if self._running:
            self.disconnect()
        self._running = True
        self.start_time = time.time()
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
        if self._pipe_path and not _IS_WINDOWS:
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

    # --- setup --------------------------------------------------------------

    def _setup(self, serial: str):
        missing = [t for t in ("scrcpy", "ffmpeg") if shutil.which(t) is None]
        if missing:
            print(f"[Scrcpy] missing tools not on PATH: {missing}")
            self.disconnect()
            return

        pipe_path = _make_pipe_path()
        self._pipe_path = pipe_path

        if _IS_WINDOWS:
            self._setup_windows(serial, pipe_path)
        else:
            self._setup_posix(serial, pipe_path)

    def _setup_posix(self, serial: str, pipe_path: str):
        os.mkfifo(pipe_path)

        video_r, video_w = os.pipe()
        audio_r, audio_w = os.pipe()

        scrcpy_proc = self._spawn_scrcpy(serial, pipe_path)
        ffmpeg_proc = self._spawn_ffmpeg_demux(pipe_path, video_w, audio_w)
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

    def _setup_windows(self, serial: str, pipe_path: str):
        import win32pipe  # windows-only optional dependency

        pipe_handle = win32pipe.CreateNamedPipe(
            pipe_path,
            win32pipe.PIPE_ACCESS_INBOUND,
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
            1,
            _RELAY_CHUNK,
            _RELAY_CHUNK,
            0,
            None,
        )

        scrcpy_proc = self._spawn_scrcpy(serial, pipe_path)
        video_ffmpeg = self._spawn_ffmpeg_stream("video")
        audio_ffmpeg = self._spawn_ffmpeg_stream("audio")
        self._procs = [scrcpy_proc, video_ffmpeg, audio_ffmpeg]

        self._threads = [
            threading.Thread(
                target=self._relay_windows_pipe,
                args=(
                    pipe_handle,
                    scrcpy_proc,
                    video_ffmpeg.stdin,
                    audio_ffmpeg.stdin,
                ),
                daemon=True,
            ),
            threading.Thread(
                target=self._read_video, args=(video_ffmpeg.stdout,), daemon=True
            ),
            threading.Thread(
                target=self._read_audio, args=(audio_ffmpeg.stdout,), daemon=True
            ),
        ]
        for t in self._threads:
            t.start()
        print(f"[Scrcpy] Streaming from {serial}")

    def _relay_windows_pipe(self, handle, scrcpy_proc, video_stdin, audio_stdin):
        import pywintypes  # windows-only
        import win32file  # windows-only
        import win32pipe  # windows-only

        try:
            win32pipe.ConnectNamedPipe(handle, None)
            while self._running and scrcpy_proc.poll() is None:
                try:
                    _, data = win32file.ReadFile(handle, _RELAY_CHUNK)
                except pywintypes.error:
                    break
                if not data:
                    break
                try:
                    video_stdin.write(data)
                    audio_stdin.write(data)
                except OSError:
                    break
        finally:
            try:
                win32file.CloseHandle(handle)
            except Exception:
                pass
            for stream in (video_stdin, audio_stdin):
                try:
                    stream.close()
                except Exception:
                    pass

    def _spawn_scrcpy(self, serial: str, pipe_path: str) -> subprocess.Popen:
        cmd = [
            "scrcpy",
            "-s",
            serial,
            "--force-adb-forward",
            "--video-source=camera",
            "--camera-facing=back",
            f"--camera-fps={ANDROID_CAMERA_FPS}",
            f"--max-size={max(GLASSES_VIDEO_WIDTH, GLASSES_VIDEO_HEIGHT)}",
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

    def _spawn_ffmpeg_demux(
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
            f"transpose=2,scale={GLASSES_VIDEO_WIDTH}:{GLASSES_VIDEO_HEIGHT}",
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

    def _spawn_ffmpeg_stream(self, stream: str) -> subprocess.Popen:
        if stream == "video":
            output_args = [
                "-map",
                "0:v",
                "-vf",
                f"transpose=2,scale={GLASSES_VIDEO_WIDTH}:{GLASSES_VIDEO_HEIGHT}",
                "-r",
                str(ANDROID_CAMERA_FPS),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
            ]
        else:
            output_args = [
                "-map",
                "0:a",
                "-f",
                "s16le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
            ]
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
            "pipe:0",
            *output_args,
            "pipe:1",
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

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
            now = time.time()
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                (GLASSES_VIDEO_HEIGHT, GLASSES_VIDEO_WIDTH, 3)
            )
            frame = FrameData(
                seq=seq,
                timestamp=int((now - self.start_time) * 1000),
                width=GLASSES_VIDEO_WIDTH,
                height=GLASSES_VIDEO_HEIGHT,
                data=arr.copy(),
            )
            with self._frame_lock:
                self._frames.append(frame)
            if seq == 0:
                print("[Scrcpy] first video frame received")
            seq += 1
        print("[Scrcpy] Video stream ended")

    def _read_audio(self, stream):
        seq = 0
        first_chunk_offset_ms = 0
        while self._running:
            raw = self._recv_exact(stream, self._CHUNK_BYTES)
            if raw is None:
                break
            now = time.time()
            audio = np.frombuffer(raw, dtype=np.int16).copy()
            if seq == 0:
                first_chunk_offset_ms = (
                    int((now - self.start_time) * 1000) - self._MS_PER_CHUNK
                )
            ts_start = first_chunk_offset_ms + seq * self._MS_PER_CHUNK
            chunk = AudioChunk(
                seq=seq,
                timestamp_start=ts_start,
                timestamp_end=ts_start + self._MS_PER_CHUNK,
                sample_rate=SAMPLE_RATE,
                num_samples=AUDIO_CHUNK_SAMPLES,
                data=audio,
            )
            with self._chunks_lock:
                self._chunks.append(chunk)
            if seq == 0:
                print("[Scrcpy] first audio chunk received")
            seq += 1
        print("[Scrcpy] Audio stream ended")
