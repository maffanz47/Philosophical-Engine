import logging
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/processed/philosophy_corpus.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_PLOT_PATH = REPORTS_DIR / "timeseries_forecast.png"

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        logger.warning(f"Data file not found at {DATA_PATH}.")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

def analyze_trends():
    df = load_data()
    if df.empty or 'year' not in df.columns:
        logger.error("Insufficient data for timeseries analysis.")
        return

    # Filter out very old or invalid years for reliable time series (e.g. keep >= 1700)
    ts_df = df[(df['year'] >= 1700) & (df['year'] <= pd.Timestamp.now().year)].copy()
    if ts_df.empty:
        logger.warning("No data after year 1700.")
        return

    # 1. Forecast monthly publication volume
    logger.info("Training Prophet for publication volume...")
    
    # Aggregate decade counts
    ts_df['decade_dt'] = pd.to_datetime(ts_df['decade'].astype(str) + '-01-01')
    volume_by_decade = ts_df.groupby('decade_dt').size().reset_index(name='count')
    
    # Downsample to monthly by interpolation
    volume_by_decade = volume_by_decade.set_index('decade_dt')
    # Resample to monthly ('MS' is month start) and interpolate
    monthly_vol = volume_by_decade.resample('MS').interpolate(method='linear').reset_index()
    monthly_vol.columns = ['ds', 'y']
    
    prophet = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    prophet.fit(monthly_vol)
    
    future = prophet.make_future_dataframe(periods=24, freq='M')
    forecast = prophet.predict(future)
    
    # Save Forecast Plot
    fig = prophet.plot(forecast)
    plt.title("Publication Volume Forecast (24 months)")
    plt.xlabel("Year")
    plt.ylabel("Interpolated Publication Volume")
    plt.savefig(FORECAST_PLOT_PATH)
    plt.close(fig)
    logger.info(f"Saved forecast plot to {FORECAST_PLOT_PATH}")

    # 2. Rolling 10-year average sentiment polarity (using decades)
    logger.info("Analyzing rolling sentiment...")
    sentiment_by_decade = ts_df.groupby('decade')['sentiment_polarity'].mean().reset_index()
    sentiment_by_decade = sentiment_by_decade.sort_values('decade')
    
    # 3. Detect anomaly decades
    logger.info("Detecting anomalies with IsolationForest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    # Fit on sentiment polarity
    X = sentiment_by_decade[['sentiment_polarity']].values
    sentiment_by_decade['anomaly'] = iso_forest.fit_predict(X)
    
    # Save processed sentiment data for the API
    sentiment_data = []
    
    for _, row in sentiment_by_decade.iterrows():
        sentiment_data.append({
            "decade": int(row['decade']),
            "avg_sentiment": round(row['sentiment_polarity'], 4),
            "is_anomaly": bool(row['anomaly'] == -1)
        })
        
    # Predict next decade sentiment (simple moving average for demo)
    last_3 = sentiment_by_decade['sentiment_polarity'].tail(3).mean()
    
    result = {
        "historical": sentiment_data,
        "forecast_next_decade_sentiment": round(last_3, 4)
    }
    
    with open(REPORTS_DIR / "sentiment_trends.json", "w") as f:
        json.dump(result, f, indent=2)
        
    logger.info("Timeseries analysis complete.")

def get_timeseries_sentiment() -> Dict[str, Any]:
    json_path = REPORTS_DIR / "sentiment_trends.json"
    if not json_path.exists():
        return {"error": "Timeseries data not generated yet."}
    with open(json_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    analyze_trends()
