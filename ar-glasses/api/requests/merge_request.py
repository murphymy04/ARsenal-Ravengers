from pydantic import BaseModel


class MergeRequest(BaseModel):
    """Merge two clusters, keeping embeddings from both."""

    keep_person_id: int
    discard_person_id: int
