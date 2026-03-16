"""Shared data models for the AR glasses prototype."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from datetime import datetime


@dataclass
class BoundingBox:
    """Face bounding box in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class DetectedFace:
    """A detected face with its cropped image."""
    bbox: BoundingBox
    crop: np.ndarray        # 112x112 RGB uint8
    frame_index: int
    timestamp: float
    blur_score: float = 0.0   # Laplacian variance — higher is sharper
    is_speaking: bool = False  # True when jawOpen blendshape exceeds threshold


@dataclass
class FaceEmbedding:
    """512-dimensional face embedding vector."""
    vector: np.ndarray      # shape (512,), float32
    model_name: str


@dataclass
class IdentityMatch:
    """Result of matching a face embedding against known people."""
    person_id: Optional[int]
    name: str
    confidence: float
    is_known: bool


@dataclass
class Person:
    """A known person stored in the database."""
    person_id: int
    name: str
    embeddings: List[FaceEmbedding] = field(default_factory=list)
    thumbnail: Optional[np.ndarray] = None  # BGR image
    notes: str = ""
    is_labeled: bool = True   # False = auto-discovered cluster awaiting a real name
    created_at: Optional[datetime] = None
    last_seen: Optional[datetime] = None


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    text: str
    start_time: float
    end_time: float
    speaker_label: Optional[str] = None
    person_id: Optional[int] = None
