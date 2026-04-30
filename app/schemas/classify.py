from pydantic import BaseModel


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    school: str
    confidence: float
    top3: list[str]
