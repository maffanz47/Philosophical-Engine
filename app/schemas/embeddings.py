from pydantic import BaseModel


class EmbeddingsMapItem(BaseModel):
    id: str
    title: str
    x: float
    y: float
    school: str
    era: str
