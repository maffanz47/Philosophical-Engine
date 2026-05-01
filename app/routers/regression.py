from fastapi import APIRouter, HTTPException
from app.schemas.regression import RegressionRequest, RegressionResponse
from ml.regression.influence_predictor import predict_influence

router = APIRouter(prefix="/regression", tags=["Regression"])

@router.post("/", response_model=RegressionResponse)
async def predict_influence_score(request: RegressionRequest):
    """Predict the influence score (normalized log of downloads) of a text."""
    try:
        score = predict_influence(request.model_dump())
        return {"influence_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
