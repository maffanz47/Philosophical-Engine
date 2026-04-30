from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SentimentForecastItem:
    decade: int
    avg_sentiment: float
    forecast_next_decade: float


def sentiment_forecast() -> list[dict[str, Any]]:
    """
    Sentiment forecast (Phase 1 placeholder).

    Phase 2+ will:
      - aggregate decade-level sentiment polarity
      - fit Facebook Prophet to forecast next 24 months
      - detect anomalies with IsolationForest
      - save plots to reports/
    """
    return []
