r"""Android camera input via scrcpy + ffmpeg (--camera android).

INMO Air 3 camera
  → scrcpy  --record=\\.\pipe\ar_camera  (Windows named pipe / POSIX FIFO)
  → relay thread  (reads pipe → writes to ffmpeg stdin)
  → ffmpeg        (H.264 → raw BGR)
  → reader thread (drains stdout, keeps latest frame)
  → frames()      (yields latest frame to pipeline)

Requirements (install on Windows, add to PATH):
  scrcpy  ≥ 2.0   https://github.com/Genymobile/scrcpy/releases
  ffmpeg          https://ffmpeg.org/download.html
  adb             bundled with scrcpy, or Android Platform Tools
  pip install pywin32          # Windows only
"""

import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Generator, Optional

import numpy as np

from config import CAMERA_WIDTH, CAMERA_HEIGHT, ANDROID_CAMERA_FPS as CAMERA_FPS

_STARTUP_TIMEOUT = 15
_CHUNK = 65536


class AndroidCamera:
    """Streams an ADB-connected Android camera via scrcpy + ffmpeg (no temp file)."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ):
        _check_dependencies()

        self._width = width
        self._height = height
        self._fps = fps
        self._frame_bytes = width * height * 3

        if sys.platform == "win32":
            pipe_path, self._pipe_handle = _create_win_pipe()
        else:
            pipe_path = _create_fifo()
            self._pipe_handle = None

        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-i",
                "pipe:0",
                "-vf",
                f"transpose=2,scale={width}:{height}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        self._scrcpy = subprocess.Popen(
            _scrcpy_args(camera_id, pipe_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._opened = True

        self._relay = threading.Thread(
            target=self._relay_loop,
            args=(pipe_path,),
            daemon=True,
        )
        self._relay.start()

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        print(f"[AndroidCamera] stream started ({width}×{height} @ {fps} fps)")

    # ------------------------------------------------------------------

    def _relay_loop(self, pipe_path: str):
        """Read from the named pipe and forward to ffmpeg stdin."""
        try:
            if sys.platform == "win32":
                _win_pipe_relay(self._pipe_handle, self._ffmpeg.stdin, self._scrcpy)
            else:
                with open(pipe_path, "rb") as f:
                    while self._opened and self._scrcpy.poll() is None:
                        chunk = f.read(_CHUNK)
                        if chunk:
                            self._ffmpeg.stdin.write(chunk)
        except Exception as e:
            print(f"[AndroidCamera] relay error: {e}")
        finally:
            try:
                self._ffmpeg.stdin.close()
            except Exception:
                pass
            if self._pipe_handle is None and os.path.exists(pipe_path):
                os.unlink(pipe_path)

    def _reader_loop(self):
        """Drain ffmpeg stdout at full speed; keep only the latest frame."""
        while self._opened:
            raw = self._ffmpeg.stdout.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                break
            frame = (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape((self._height, self._width, 3))
                .copy()
            )
            with self._lock:
                self._latest = frame
        self._opened = False

    # ------------------------------------------------------------------

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield the latest BGR frame as fast as the pipeline consumes it."""
        while self.is_opened:
            with self._lock:
                frame = self._latest
                self._latest = None
            if frame is not None:
                yield frame
            else:
                time.sleep(0.001)

    @property
    def is_opened(self) -> bool:
        return self._opened and self._scrcpy.poll() is None

    def close(self):
        self._opened = False
        for proc in (self._ffmpeg, self._scrcpy):
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        self._relay.join(timeout=3)
        self._reader.join(timeout=3)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def _create_win_pipe():
    """Create a Windows named pipe with a unique name; return (path, handle)."""
    import win32pipe

    pipe_name = rf"\\.\pipe\ar_camera_{uuid.uuid4().hex[:8]}"
    handle = win32pipe.CreateNamedPipe(
        pipe_name,
        win32pipe.PIPE_ACCESS_INBOUND,
        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
        1,
        _CHUNK,
        _CHUNK,
        0,
        None,
    )
    return pipe_name, handle


def _win_pipe_relay(handle, ffmpeg_stdin, scrcpy_proc):
    """Block until scrcpy connects, then relay pipe bytes to ffmpeg stdin."""
    import win32pipe
    import win32file
    import pywintypes

    win32pipe.ConnectNamedPipe(handle, None)
    try:
        while scrcpy_proc.poll() is None:
            try:
                _, data = win32file.ReadFile(handle, _CHUNK)
                if data:
                    ffmpeg_stdin.write(data)
            except pywintypes.error:
                break
    finally:
        win32file.CloseHandle(handle)


def _create_fifo() -> str:
    """Create a POSIX FIFO; return its path."""
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="ar_android_")
    path = os.path.join(tmpdir, "cam.mkv")
    os.mkfifo(path)
    return path


def _scrcpy_args(camera_id: int, record_path: str) -> list[str]:
    return [
        "scrcpy",
        "--video-source=camera",
        "--camera-facing=back",
        "--no-audio",
        "--no-playback",
        "--no-window",
        f"--record={record_path}",
        "--record-format=mkv",
    ]


def _check_dependencies():
    missing = [cmd for cmd in ("scrcpy", "ffmpeg", "adb") if shutil.which(cmd) is None]
    if missing:
        raise RuntimeError(
            f"Missing required tools: {', '.join(missing)}\n"
            "  scrcpy: https://github.com/Genymobile/scrcpy/releases\n"
            "  ffmpeg: https://ffmpeg.org/download.html\n"
            "  adb:    bundled with scrcpy or Android Platform Tools"
        )
    if sys.platform == "win32":
        try:
            import win32pipe  # noqa: F401
        except ImportError:
            raise RuntimeError("pywin32 not installed. Run: pip install pywin32")
