from fastapi import APIRouter
from pydantic import BaseModel

from ml.timeseries.trend_analyzer import sentiment_forecast as sentiment_forecast_stub

router = APIRouter(prefix="/timeseries", tags=["timeseries"])


class SentimentForecastItem(BaseModel):
    decade: int
    avg_sentiment: float
    forecast_next_decade: float


@router.get("/sentiment", response_model=list[SentimentForecastItem])
def sentiment_forecast() -> list[SentimentForecastItem]:
    """
    Phase 1:
      - Returns forecast items when artifacts exist (currently stub returns []).
    """
    results = sentiment_forecast_stub()

    items: list[SentimentForecastItem] = []
    for r in results:
        if isinstance(r, dict):
            items.append(SentimentForecastItem(**r))
    return items
