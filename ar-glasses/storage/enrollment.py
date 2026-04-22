"""Enrollment workflow for adding new people.

Captures face embeddings and stores them with a name in the database.
Supports enrolling from live camera or from image files.
"""

import cv2
import numpy as np
from typing import List, Optional

from models import FaceEmbedding
from storage.database import Database
from processing.face_detector import FaceDetector
from processing.face_embedder import FaceEmbedder


class Enrollment:
    """Manages adding new people to the recognition database."""

    def __init__(self, db: Database, detector: FaceDetector, embedder: FaceEmbedder):
        self._db = db
        self._detector = detector
        self._embedder = embedder

    def enroll_from_image(
        self, name: str, image: np.ndarray, notes: str = ""
    ) -> Optional[int]:
        """Enroll a person from a single BGR image.

        Args:
            name: person's name.
            image: BGR image containing their face.
            notes: optional notes about the person.

        Returns:
            person_id if successful, None if no face detected.
        """
        faces = self._detector.detect(image)
        if not faces:
            return None

        face = max(faces, key=lambda f: f.blur_score)
        embedding = self._embedder.embed(face.crop)

        # Create thumbnail from the crop (convert RGB crop to BGR for storage)
        thumbnail = cv2.cvtColor(face.crop, cv2.COLOR_RGB2BGR)

        person_id = self._db.add_person(name, notes=notes, thumbnail=thumbnail)
        self._db.add_embedding(person_id, embedding)
        return person_id

    def enroll_from_images(
        self, name: str, images: List[np.ndarray], notes: str = ""
    ) -> Optional[int]:
        """Enroll a person from multiple BGR images for better accuracy.

        Uses the first successful detection as the thumbnail, and stores
        all extracted embeddings.

        Returns:
            person_id if at least one face was detected, None otherwise.
        """
        embeddings: List[FaceEmbedding] = []
        best_face = None

        for img in images:
            faces = self._detector.detect(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: f.bbox.width * f.bbox.height)
            embeddings.append(self._embedder.embed(face.crop))
            if best_face is None or face.blur_score > best_face.blur_score:
                best_face = face

        thumbnail = (
            cv2.cvtColor(best_face.crop, cv2.COLOR_RGB2BGR) if best_face else None
        )

        if not embeddings:
            return None

        person_id = self._db.add_person(name, notes=notes, thumbnail=thumbnail)
        for emb in embeddings:
            self._db.add_embedding(person_id, emb)
        return person_id

    def add_embedding_to_person(self, person_id: int, image: np.ndarray) -> bool:
        """Add an additional embedding to an existing person.

        Returns True if a face was detected and embedding added.
        """
        faces = self._detector.detect(image)
        if not faces:
            return False
        face = max(faces, key=lambda f: f.bbox.width * f.bbox.height)
        embedding = self._embedder.embed(face.crop)
        self._db.add_embedding(person_id, embedding)
        return True
