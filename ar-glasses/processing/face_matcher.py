"""Face matching via cosine similarity against per-person mean embeddings.

Instead of comparing a query against every stored embedding, we precompute
a single L2-normalised mean embedding per person when the gallery is loaded
(#4).  This is O(n_people) per query rather than O(n_people * n_embeddings),
and is more robust to individual outlier embeddings.
"""

import numpy as np
from models import FaceEmbedding, IdentityMatch, Person

from config import MATCH_THRESHOLD, UNKNOWN_LABEL


class FaceMatcher:
    """Matches a face embedding against a gallery of known people."""

    def __init__(self, threshold: float = MATCH_THRESHOLD):
        self.threshold = threshold
        self._gallery: list[Person] = []
        # Precomputed (person, normalised_mean_vector) pairs — rebuilt on every
        # update_gallery() call so matching is a simple dot-product loop.
        self._means: list[tuple[Person, np.ndarray]] = []

    def update_gallery(self, people: list[Person]):
        """Replace the gallery and recompute per-person mean embeddings."""
        self._gallery = people
        self._means = []
        for person in people:
            if not person.embeddings:
                continue
            vecs = np.array([e.vector for e in person.embeddings], dtype=np.float32)
            mean = vecs.mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm > 0:
                self._means.append((person, mean / norm))

    def match(self, embedding: FaceEmbedding) -> IdentityMatch:
        """Return the best matching person, or an unknown result.

        Args:
            embedding: query face embedding.

        Returns:
            IdentityMatch — is_known=True if the best score >= threshold.
        """
        if not self._means:
            return IdentityMatch(
                person_id=None,
                name=UNKNOWN_LABEL,
                confidence=0.0,
                is_known=False,
            )

        query = embedding.vector
        qnorm = np.linalg.norm(query)
        if qnorm == 0:
            return IdentityMatch(
                person_id=None,
                name=UNKNOWN_LABEL,
                confidence=0.0,
                is_known=False,
            )

        query_n = query / qnorm
        best_score = -1.0
        best_person = None

        for person, mean_n in self._means:
            score = float(np.dot(query_n, mean_n))
            if score > best_score:
                best_score = score
                best_person = person

        if best_person is not None and best_score >= self.threshold:
            return IdentityMatch(
                person_id=best_person.person_id,
                name=best_person.name,
                confidence=best_score,
                is_known=True,
            )

        return IdentityMatch(
            person_id=None,
            name=UNKNOWN_LABEL,
            confidence=max(best_score, 0.0),
            is_known=False,
        )

    def rank_candidates(self, embedding: FaceEmbedding) -> list[tuple[Person, float]]:
        """Return all gallery people sorted by cosine similarity (descending).

        Args:
            embedding: query face embedding.

        Returns:
            List of (Person, score) pairs, highest score first.
        """
        if not self._means:
            return []

        query = embedding.vector
        qnorm = np.linalg.norm(query)
        if qnorm == 0:
            return []

        query_n = query / qnorm
        scores = [
            (person, float(np.dot(query_n, mean_n))) for person, mean_n in self._means
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
