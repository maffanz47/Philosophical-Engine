from fastapi import APIRouter

from app.schemas.regression import InfluencePredictRequest, InfluencePredictResponse
from ml.regression.influence_predictor import predict_influence

router = APIRouter(prefix="/regression", tags=["regression"])


@router.post("/influence", response_model=InfluencePredictResponse)
def predict_influence_endpoint(body: InfluencePredictRequest) -> InfluencePredictResponse:
    """
    Phase 1:
      - If no trained artifact exists, returns 0.0.
      - Otherwise uses the latest saved regression model artifact.

    Phase 2+ will construct the full feature vector (structured + TF-IDF).
    """
    score = predict_influence({"text": body.text})
    return InfluencePredictResponse(influence_score=score)
