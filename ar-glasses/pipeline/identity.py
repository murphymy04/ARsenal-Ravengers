"""Identity module implementations for the diarization pipeline.

This was previosly coupled in main.py. I wanted to rebuild this incrementally
to make sure it worked. It works and now acts as a reusable component.

Now that it is a module, it is also much easier to optimize it.

NullIdentity returns unknown for every face (zero dependencies).
FullIdentity wraps FaceEmbedder + FaceMatcher + Database for real recognition
with automatic pending-cluster promotion.
"""

import cv2
import numpy as np

from config import (
    EMBEDDING_UPDATE_INTERVAL,
    MAX_EMBEDDINGS_PER_PERSON,
    EMBEDDING_DIVERSITY_THRESHOLD,
    FACE_BLUR_THRESHOLD,
    MIN_SIGHTINGS_TO_CLUSTER,
    PENDING_CLUSTER_SIMILARITY,
    PENDING_EXPIRY_FRAMES,
)
from models import DetectedFace, FaceEmbedding, IdentityMatch

class NullIdentity:
    def identify(self, face: DetectedFace, frame_count: int) -> IdentityMatch:
        return IdentityMatch(person_id=None, name="unknown", confidence=0.0, is_known=False)

# This is exactly the same code as in main.py
# Just putting it into a separate file

class FullIdentity:
    def __init__(self, embedder, matcher, db):
        self._embedder = embedder
        self._matcher = matcher
        self._db = db

        self._pending: list[dict] = []
        self._last_embedding_update: dict[int, int] = {}

        people = db.get_all_people()
        matcher.update_gallery(people)

    def identify(self, face: DetectedFace, frame_count: int) -> IdentityMatch:
        self._expire_pending(frame_count)

        embedding = self._embedder.embed(face.crop)
        match = self._matcher.match(embedding)

        if match.is_known:
            self._maybe_store_embedding(match.person_id, embedding, face, frame_count)
            return match

        match, promoted = self._update_pending(embedding, face, frame_count)
        if promoted:
            self._matcher.update_gallery(self._db.get_all_people())
        return match

    def _expire_pending(self, frame_count: int):
        self._pending = [
            p for p in self._pending
            if frame_count - p["last_frame"] < PENDING_EXPIRY_FRAMES
        ]

    def _maybe_store_embedding(
        self, person_id: int, embedding: FaceEmbedding, face: DetectedFace, frame_count: int,
    ) -> bool:
        self._db.update_last_seen(person_id)
        if frame_count - self._last_embedding_update.get(person_id, 0) < EMBEDDING_UPDATE_INTERVAL:
            return False

        person = self._db.get_person(person_id)
        self._last_embedding_update[person_id] = frame_count

        if (
            person
            and len(person.embeddings) < MAX_EMBEDDINGS_PER_PERSON
            and face.blur_score >= FACE_BLUR_THRESHOLD
            and _is_diverse(embedding, person.embeddings, EMBEDDING_DIVERSITY_THRESHOLD)
        ):
            self._db.add_embedding(person_id, embedding)
            self._matcher.update_gallery(self._db.get_all_people())
            return True

        return False

    def _update_pending(
        self, embedding: FaceEmbedding, face: DetectedFace, frame_count: int,
    ) -> tuple[IdentityMatch, bool]:
        best_idx, best_score = -1, -1.0
        for i, pc in enumerate(self._pending):
            score = _cosine_sim(embedding.vector, pc["mean"])
            if score > best_score:
                best_score, best_idx = score, i

        if best_idx >= 0 and best_score >= PENDING_CLUSTER_SIMILARITY:
            pc = self._pending[best_idx]
            pc["embeddings"].append(embedding)
            pc["last_frame"] = frame_count
            n = len(pc["embeddings"])
            pc["mean"] = pc["mean"] + (embedding.vector - pc["mean"]) / n

            if n >= MIN_SIGHTINGS_TO_CLUSTER:
                person_id, auto_name = self._db.add_auto_person(thumbnail=pc["thumbnail"])
                for emb in pc["embeddings"]:
                    self._db.add_embedding(person_id, emb)
                self._last_embedding_update[person_id] = frame_count
                self._pending.pop(best_idx)
                return IdentityMatch(
                    person_id=person_id, name=auto_name, confidence=1.0, is_known=True,
                ), True

            return IdentityMatch(person_id=None, name="Unknown", confidence=0.0, is_known=False), False

        self._pending.append({
            "embeddings": [embedding],
            "mean": embedding.vector.copy(),
            "thumbnail": cv2.cvtColor(face.crop, cv2.COLOR_RGB2BGR),
            "last_frame": frame_count,
        })
        return IdentityMatch(person_id=None, name="Unknown", confidence=0.0, is_known=False), False


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _is_diverse(new_emb: FaceEmbedding, existing: list[FaceEmbedding], min_distance: float) -> bool:
    return all(
        _cosine_sim(new_emb.vector, e.vector) < 1.0 - min_distance
        for e in existing
    )
