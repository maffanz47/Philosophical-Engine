from fastapi import APIRouter, HTTPException
from app.schemas.classify import ClassificationRequest, ClassificationResponse
from ml.classification.school_classifier import predict_school

router = APIRouter(prefix="/classify", tags=["Classification"])

@router.post("/", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """Classify a text passage into a philosophical school."""
    try:
        features = [request.avg_sentence_length, request.vocab_richness, request.sentiment_polarity]
        result = predict_school(request.text, features)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
