from pydantic import BaseModel


class UnlabeledResponse(BaseModel):
    """Unlabeled cluster awaiting a name assignment."""

    person_id: int
    name: str
    embedding_count: int
    last_seen: str | None
    thumbnail: str | None  # base64-encoded JPEG
