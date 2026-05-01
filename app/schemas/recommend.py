from typing import List, Optional
from pydantic import BaseModel

class RecommendationResponseItem(BaseModel):
    gutenberg_id: int
    title: str
    author: str
    similarity: float
    reason: str

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationResponseItem]
