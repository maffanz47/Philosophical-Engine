import os
import time
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.sparse import hstack

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path("data/processed/philosophy_corpus.csv")
MODEL_DIR = Path("models/regression")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_EXPERIMENT = "philosophical-engine-regression"

# Global loaded model for inference
_best_model = None
_vectorizer = None
_era_encoder = None


def _latest_model_path(subdir: str) -> Optional[str]:
    model_dir = Path("models") / subdir
    if not model_dir.exists():
        return None

    candidates = list(model_dir.glob("influence_predictor_v*.pkl"))
    if not candidates:
        return None

    return str(max(candidates, key=os.path.getctime))

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        logger.warning(f"Data file not found at {DATA_PATH}.")
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)

def plot_actual_vs_predicted(y_true, y_pred, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--') # Identity line
    plt.xlabel('Actual Influence Score')
    plt.ylabel('Predicted Influence Score')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_models():
    df = load_data()
    if df.empty or 'download_count' not in df.columns:
        logger.error("Insufficient data to train regression models.")
        return

    # Drop missing text
    df = df.dropna(subset=['full_text'])
    
    # Calculate influence_score = log1p(download_count) and normalize to [0,1]
    raw_scores = np.log1p(df['download_count'])
    scaler = MinMaxScaler()
    df['influence_score'] = scaler.fit_transform(raw_scores.values.reshape(-1, 1))

    # Features: avg_sentence_length, vocab_richness, sentiment_polarity, named_entity_count, era encoded, text length, TF-IDF (top 500)
    numeric_cols = ['avg_sentence_length', 'vocab_richness', 'sentiment_polarity', 'named_entity_count', 'text_length']
    X_num = df[numeric_cols].fillna(0).values
    
    # Encode Era
    era_le = LabelEncoder()
    df['era_encoded'] = era_le.fit_transform(df['era_label'].fillna("Unknown"))
    X_era = df[['era_encoded']].values
    
    X_structured = np.hstack((X_num, X_era))
    X_text = df['full_text'].astype(str).tolist()
    y = df['influence_score'].values
    
    X_train_text, X_test_text, X_train_struct, X_test_struct, y_train, y_test = train_test_split(
        X_text, X_structured, y, test_size=0.2, random_state=42
    )

    # TF-IDF Vectorization (Top 500)
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    X_train_combined = hstack([X_train_tfidf, X_train_struct]).tocsr()
    X_test_combined = hstack([X_test_tfidf, X_test_struct]).tocsr()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    best_r2 = -float("inf")
    best_model_name = ""
    timestamp = int(time.time())

    # 1. Baseline: Ridge Regression
    with mlflow.start_run(run_name=f"Ridge_{timestamp}"):
        logger.info("Training Ridge Regression...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_combined, y_train)
        
        preds = ridge.predict(X_test_combined)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_params({"alpha": 1.0, "model": "Ridge"})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        
        plot_path = "reports/ridge_scatter.png"
        plot_actual_vs_predicted(y_test, preds, "Ridge Regression: Actual vs Predicted", plot_path)
        mlflow.log_artifact(plot_path)
        
        mlflow.sklearn.log_model(ridge, artifact_path="model")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = "Ridge"
            joblib.dump({"model": ridge, "vectorizer": tfidf, "era_encoder": era_le}, 
                        MODEL_DIR / f"influence_predictor_v{timestamp}.pkl")

    # 2. Improved: Gradient Boosting Regressor
    with mlflow.start_run(run_name=f"GBR_{timestamp}"):
        logger.info("Training Gradient Boosting Regressor...")
        gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gbr.fit(X_train_combined, y_train)
        
        preds = gbr.predict(X_test_combined)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_params({"n_estimators": 100, "model": "GradientBoosting"})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        
        plot_path = "reports/gbr_scatter.png"
        plot_actual_vs_predicted(y_test, preds, "Gradient Boosting: Actual vs Predicted", plot_path)
        mlflow.log_artifact(plot_path)
        
        mlflow.sklearn.log_model(gbr, artifact_path="model")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = "GradientBoosting"
            joblib.dump({"model": gbr, "vectorizer": tfidf, "era_encoder": era_le}, 
                        MODEL_DIR / f"influence_predictor_v{timestamp}.pkl")

    # 3. Advanced: LightGBM Regressor with early stopping
    with mlflow.start_run(run_name=f"LightGBM_{timestamp}"):
        logger.info("Training LightGBM Regressor...")
        lgb_train = lgb.Dataset(X_train_combined, y_train)
        lgb_eval = lgb.Dataset(X_test_combined, y_test, reference=lgb_train)
        
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=500,
                        valid_sets=[lgb_train, lgb_eval],
                        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
                        
        preds = gbm.predict(X_test_combined, num_iteration=gbm.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        mlflow.log_params(params)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        
        plot_path = "reports/lgb_scatter.png"
        plot_actual_vs_predicted(y_test, preds, "LightGBM: Actual vs Predicted", plot_path)
        mlflow.log_artifact(plot_path)
        
        mlflow.lightgbm.log_model(gbm, artifact_path="model")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = "LightGBM"
            joblib.dump({"model": gbm, "vectorizer": tfidf, "era_encoder": era_le}, 
                        MODEL_DIR / f"influence_predictor_v{timestamp}.pkl")

    logger.info(f"Training complete. Best model: {best_model_name} with R2: {best_r2:.4f}")

def load_best_model():
    global _best_model, _vectorizer, _era_encoder
    if _best_model is not None:
        return

    latest_model_path = _latest_model_path("regression")
    if not latest_model_path:
        logger.warning("No trained regression model found.")
        return

    try:
        data = joblib.load(latest_model_path)
    except Exception as e:
        logger.error(f"Failed to load regression model from {latest_model_path}: {e}")
        return

    _best_model = data.get('model')
    _vectorizer = data.get('vectorizer')
    _era_encoder = data.get('era_encoder')

def predict_influence(features: Dict[str, Any]) -> float:
    """
    Predict influence score in [0, 1].
    Features dict should contain: full_text, avg_sentence_length, vocab_richness,
    sentiment_polarity, named_entity_count, era_label, text_length.
    """
    load_best_model()
    if _best_model is None:
        return 0.0
        
    try:
        text = features.get('full_text', '')
        num_feats = [
            features.get('avg_sentence_length', 0.0),
            features.get('vocab_richness', 0.0),
            features.get('sentiment_polarity', 0.0),
            features.get('named_entity_count', 0),
            features.get('text_length', 0)
        ]
        
        era = features.get('era_label', 'Unknown')
        # Handle unseen eras gracefully
        try:
            era_encoded = _era_encoder.transform([era])[0]
        except ValueError:
            era_encoded = 0 # Default fallback
            
        X_struct = np.array([num_feats + [era_encoded]])
        X_tfidf = _vectorizer.transform([text])
        
        X_combined = hstack([X_tfidf, X_struct]).tocsr()
        
        if isinstance(_best_model, lgb.Booster):
            pred = _best_model.predict(X_combined)[0]
        else:
            pred = _best_model.predict(X_combined)[0]
            
        # Clip to [0,1]
        return float(np.clip(pred, 0.0, 1.0))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 0.0

if __name__ == "__main__":
    train_models()
