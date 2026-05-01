from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from ml.timeseries.trend_analyzer import get_timeseries_sentiment

router = APIRouter(prefix="/timeseries", tags=["Time Series"])

@router.get("/sentiment", response_model=Dict[str, Any])
async def get_sentiment_trends():
    """Returns decade-level sentiment trends and anomalies."""
    try:
        data = get_timeseries_sentiment()
        if "error" in data:
            raise HTTPException(status_code=500, detail=data["error"])
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
