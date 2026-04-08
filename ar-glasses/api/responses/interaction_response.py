from pydantic import BaseModel


class InteractionResponse(BaseModel):
    """An interaction record as returned by the API."""

    interaction_id: int
    person_id: int | None
    timestamp: str
    transcript: str
    context: str

    class Config:
        from_attributes = True
