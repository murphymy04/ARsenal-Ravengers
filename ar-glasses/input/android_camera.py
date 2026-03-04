"""Android camera input via scrcpy + ffmpeg using a named FIFO.

scrcpy does not flush its MKV output when writing to an anonymous stdout pipe,
so we use a temporary named FIFO as the bridge instead.
"""

import os
import subprocess
import tempfile
import numpy as np
from typing import Generator, Optional

from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class AndroidCamera:
    """Streams an Android device camera via scrcpy → FIFO → ffmpeg.

    Requires scrcpy and ffmpeg to be installed and a device connected via ADB.
    Yields BGR frames at the configured resolution, matching the Camera interface.
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ):
        self._width = width
        self._height = height
        self._frame_bytes = width * height * 3

        # Create a named FIFO in a temp dir; scrcpy writes MKV here,
        # ffmpeg reads from it. Anonymous pipes don't work because scrcpy
        # buffers internally and never flushes to a non-seekable pipe.
        self._fifo_dir = tempfile.mkdtemp()
        self._fifo_path = os.path.join(self._fifo_dir, "cam.mkv")
        os.mkfifo(self._fifo_path)

        self._scrcpy = subprocess.Popen(
            [
                "scrcpy",
                "--video-source=camera",
                f"--camera-id={camera_id}",
                "--no-audio",
                "--no-playback",
                f"--record={self._fifo_path}",
                "--record-format=mkv",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-i", self._fifo_path,
                "-vf", f"scale={width}:{height}",
                "-r", str(fps),
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        self._opened = True

    def read(self) -> Optional[np.ndarray]:
        """Read a single BGR frame. Returns None if stream ends."""
        raw = self._ffmpeg.stdout.read(self._frame_bytes)
        if len(raw) < self._frame_bytes:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((self._height, self._width, 3)).copy()

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield BGR frames until the stream closes."""
        while self.is_opened:
            frame = self.read()
            if frame is None:
                break
            yield frame

    @property
    def is_opened(self) -> bool:
        return self._opened and self._ffmpeg.poll() is None

    def close(self):
        self._opened = False
        self._ffmpeg.terminate()
        self._scrcpy.terminate()
        try:
            os.unlink(self._fifo_path)
            os.rmdir(self._fifo_dir)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
