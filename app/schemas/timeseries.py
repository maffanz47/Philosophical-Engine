from pydantic import BaseModel


class SentimentTrendItem(BaseModel):
    decade: int
    avg_sentiment: float
    forecast_next_decade: float | None = None


class TimeseriesResponse(BaseModel):
    trends: list[SentimentTrendItem]


class ForecastResponse(BaseModel):
    forecast: list[SentimentTrendItem]
