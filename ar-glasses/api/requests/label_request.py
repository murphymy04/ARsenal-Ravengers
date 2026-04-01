from pydantic import BaseModel


class LabelRequest(BaseModel):
    """Assign a name to an unlabeled cluster."""
    name: str
