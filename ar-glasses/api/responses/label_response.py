from pydantic import BaseModel


class LabelResponse(BaseModel):
    """Response after labeling or merging."""

    person_id: int
    name: str
    is_labeled: bool
    action: str
    details: str | None = None
