from pydantic import BaseModel


class NotesRequest(BaseModel):
    """Update notes for a person."""

    notes: str
