"""Face matching via cosine similarity.

Compares a face embedding against a gallery of known people
and returns the best match or an unknown result.
"""

import numpy as np
from typing import List

from models import FaceEmbedding, IdentityMatch, Person
from config import MATCH_THRESHOLD, UNKNOWN_LABEL


class FaceMatcher:
    """Matches face embeddings against known people using cosine similarity."""

    def __init__(self, threshold: float = MATCH_THRESHOLD):
        self.threshold = threshold
        self._gallery: List[Person] = []

    def update_gallery(self, people: List[Person]):
        """Replace the gallery of known people.

        Args:
            people: list of Person objects with embeddings.
        """
        self._gallery = people

    def match(self, embedding: FaceEmbedding) -> IdentityMatch:
        """Find the best matching person for an embedding.

        Args:
            embedding: the query face embedding.

        Returns:
            IdentityMatch with the best match or unknown.
        """
        if not self._gallery:
            return IdentityMatch(
                person_id=None,
                name=UNKNOWN_LABEL,
                confidence=0.0,
                is_known=False,
            )

        best_score = -1.0
        best_person = None

        query = embedding.vector
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return IdentityMatch(
                person_id=None, name=UNKNOWN_LABEL,
                confidence=0.0, is_known=False,
            )

        for person in self._gallery:
            for emb in person.embeddings:
                ref = emb.vector
                ref_norm = np.linalg.norm(ref)
                if ref_norm == 0:
                    continue
                score = float(np.dot(query, ref) / (query_norm * ref_norm))
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
            confidence=best_score if best_score > 0 else 0.0,
            is_known=False,
        )
