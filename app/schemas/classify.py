from typing import Optional, List
from pydantic import BaseModel, Field

class ClassificationRequest(BaseModel):
    text: str = Field(..., description="Text passage to classify")
    avg_sentence_length: Optional[float] = 0.0
    vocab_richness: Optional[float] = 0.0
    sentiment_polarity: Optional[float] = 0.0

class SchoolProbability(BaseModel):
    school: str
    confidence: float

class ClassificationResponse(BaseModel):
    school: str
    confidence: float
    top3: List[SchoolProbability]
