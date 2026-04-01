from pydantic import BaseModel
from typing import Optional


class PersonResponse(BaseModel):
    """A person as returned by the API (includes metadata, not full embeddings)."""
    person_id: int
    name: str
    is_labeled: bool
    embedding_count: int
    notes: str
    created_at: str
    last_seen: Optional[str]
    thumbnail_url: Optional[str] = None

    class Config:
        from_attributes = True
