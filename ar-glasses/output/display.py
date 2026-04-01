"""Display overlay for face recognition results.

Draws bounding boxes, names, and confidence scores on frames
using OpenCV drawing primitives.
"""

import cv2
import numpy as np
from typing import List

from models import DetectedFace, IdentityMatch
from config import (
    BBOX_COLOR, UNKNOWN_BBOX_COLOR, TEXT_COLOR,
    FONT_SCALE, FONT_THICKNESS,
)

_SPEAKING_COLOR = (0, 220, 255)  # yellow-ish — distinct from green/red bbox colors


class Display:
    """Renders face recognition overlays on video frames."""

    def __init__(self, window_name: str = "AR Glasses Prototype"):
        self.window_name = window_name

    def draw(
        self,
        frame: np.ndarray,
        faces: List[DetectedFace],
        matches: List[IdentityMatch],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a frame.

        Args:
            frame: BGR image to annotate (modified in-place).
            faces: detected faces.
            matches: identity matches (same order as faces).

        Returns:
            Annotated BGR frame.
        """
        for face, match in zip(faces, matches):
            bbox = face.bbox
            color = BBOX_COLOR if match.is_known else UNKNOWN_BBOX_COLOR

            # Bounding box — thicker + speaking-colour outline when talking
            cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
            if face.is_speaking:
                cv2.rectangle(
                    frame,
                    (bbox.x1 - 3, bbox.y1 - 3),
                    (bbox.x2 + 3, bbox.y2 + 3),
                    _SPEAKING_COLOR, 2,
                )

            # Label: "Name (0.85)" or "Unknown", with speaking dot prefix
            if match.is_known:
                label = f"{match.name} ({match.confidence:.2f})"
            else:
                label = "Unknown"
            if face.is_speaking:
                label = "* " + label

            # Background rectangle for text readability
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
            )
            cv2.rectangle(
                frame,
                (bbox.x1, bbox.y1 - th - 10),
                (bbox.x1 + tw + 4, bbox.y1),
                color, -1,
            )
            cv2.putText(
                frame, label,
                (bbox.x1 + 2, bbox.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, TEXT_COLOR, FONT_THICKNESS,
            )

            # Speaking indicator dot in the top-right corner of the bbox
            if face.is_speaking:
                dot_x = bbox.x2 - 10
                dot_y = bbox.y1 + 10
                cv2.circle(frame, (dot_x, dot_y), 7, _SPEAKING_COLOR, -1)
                cv2.circle(frame, (dot_x, dot_y), 7, (0, 0, 0), 1)  # outline

        return frame

    def show(self, frame: np.ndarray) -> bool:
        """Display the frame. Returns False if user pressed 'q'."""
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF != ord("q")

    def close(self):
        """Destroy the display window."""
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
