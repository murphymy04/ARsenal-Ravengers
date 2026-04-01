from pydantic import BaseModel
from typing import Optional


class LabelResponse(BaseModel):
    """Response after labeling or merging."""
    person_id: int
    name: str
    is_labeled: bool
    action: str
    details: Optional[str] = None
