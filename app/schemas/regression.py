from typing import Optional
from pydantic import BaseModel, Field

class RegressionRequest(BaseModel):
    full_text: str = Field(..., description="Full text for TF-IDF extraction")
    avg_sentence_length: Optional[float] = 0.0
    vocab_richness: Optional[float] = 0.0
    sentiment_polarity: Optional[float] = 0.0
    named_entity_count: Optional[int] = 0
    text_length: Optional[int] = 0
    era_label: Optional[str] = "Unknown"

class RegressionResponse(BaseModel):
    influence_score: float = Field(..., description="Predicted influence score in [0, 1]")
