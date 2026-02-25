"""Camera input using OpenCV VideoCapture.

Provides a simple wrapper around cv2.VideoCapture with
a frame generator for the main video loop.
"""

import cv2
import numpy as np
from typing import Generator, Optional

from config import CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class Camera:
    """OpenCV camera wrapper."""

    def __init__(
        self,
        source: int | str = CAMERA_SOURCE,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ):
        self._source = source
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

    def read(self) -> Optional[np.ndarray]:
        """Read a single BGR frame. Returns None if capture fails."""
        ret, frame = self._cap.read()
        return frame if ret else None

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Yield BGR frames until the camera is closed."""
        while self._cap.isOpened():
            frame = self.read()
            if frame is None:
                break
            yield frame

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def close(self):
        """Release the camera."""
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
