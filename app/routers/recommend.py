from fastapi import APIRouter, HTTPException, Query
from app.schemas.recommend import RecommendationResponse
from ml.recommendation.recommender import recommend

router = APIRouter(prefix="/recommend", tags=["Recommendation"])

@router.get("/", response_model=RecommendationResponse)
async def get_recommendations(query: str = Query(..., description="Book ID or raw text"), n: int = 10):
    """Get top N similar books using Hybrid recommendation."""
    try:
        recs = recommend(query, n)
        if recs and "error" in recs[0]:
            raise HTTPException(status_code=500, detail=recs[0]["error"])
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
