from pydantic import BaseModel


class RecommendItem(BaseModel):
    title: str
    author: str | None = None
    similarity: float
    reason: str | None = None
