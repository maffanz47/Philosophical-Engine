from fastapi import APIRouter

from app.schemas.classify import ClassifyRequest, ClassifyResponse
from ml.classification.school_classifier import predict_school

router = APIRouter(prefix="/classify", tags=["classification"])


@router.post("/", response_model=ClassifyResponse)
def classify_text(body: ClassifyRequest) -> ClassifyResponse:
    """
    Phase 1:
      - If no trained model artifacts exist yet, returns a safe default.
      - Otherwise uses the latest saved artifact to predict school.
    """
    result = predict_school(body.text)
    return ClassifyResponse(**result)
