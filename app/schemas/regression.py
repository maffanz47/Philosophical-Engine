from pydantic import BaseModel


class InfluencePredictRequest(BaseModel):
    text: str


class InfluencePredictResponse(BaseModel):
    influence_score: float
