from pydantic import BaseModel
from typing import Optional


class UnlabeledResponse(BaseModel):
    """Unlabeled cluster awaiting a name assignment."""
    person_id: int
    name: str
    embedding_count: int
    last_seen: Optional[str]
    thumbnail_url: Optional[str]
