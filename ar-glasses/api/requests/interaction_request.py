from pydantic import BaseModel


class InteractionRequest(BaseModel):
    """Add a new interaction record."""

    person_id: int | None = None
    transcript: str
    context: str = ""
