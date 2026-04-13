from pydantic import BaseModel


class PersonResponse(BaseModel):
    """A person as returned by the API (includes metadata, not full embeddings)."""

    person_id: int
    name: str
    is_labeled: bool
    embedding_count: int
    notes: str
    created_at: str
    last_seen: str | None
    thumbnail: str | None = None

    class Config:
        from_attributes = True
