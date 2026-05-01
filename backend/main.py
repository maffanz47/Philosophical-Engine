"""
main.py
=======
FastAPI backend for the Philosophical Text Engine.

Endpoints:
  GET  /           — Health check / API info
  POST /predict    — Predict school + complexity + recommendations
                     Accepts: JSON body OR .txt file upload
  GET  /schools    — List all philosophical schools
  GET  /metrics    — Last pipeline run metrics
"""

import os
import io
import json
import logging
import pickle
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths & env
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

# Add src to path
import sys
# sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'src')))


from preprocessor import compute_complexity_score
from models import (
    BaselineANN,
    ProANN,
    get_distilbert_embeddings,
    get_recommendations,
    DEVICE,
    MODELS_DIR,
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR_PATH = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
MODEL_STATE: dict = {}


def _load_latest(prefix: str, models_dir: Path, ext: str = "pkl"):
    """Load the most recent versioned model file matching a prefix."""
    pattern = f"{prefix}_v1.0_*.{ext}"
    candidates = sorted(models_dir.glob(pattern), reverse=True)
    if not candidates:
        return None
    path = candidates[0]
    logger.info("Loading %s …", path.name)
    if ext == "pt":
        return torch.load(path, map_location=DEVICE, weights_only=True)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_models():
    """Load all trained model artifacts into MODEL_STATE."""
    state = {}

    # ── Label Encoders ──────────────────────────────────────────────────────
    le_baseline = _load_latest("LabelEncoder_Baseline", MODELS_DIR_PATH)
    le_pro      = _load_latest("LabelEncoder_Pro",      MODELS_DIR_PATH)
    state["le_baseline"] = le_baseline
    state["le_pro"]      = le_pro
    num_classes = len(le_baseline.classes_) if le_baseline else 5

    # ── TF-IDF Vectorizer ───────────────────────────────────────────────────
    tfidf = _load_latest("TFIDF_Baseline", MODELS_DIR_PATH)
    state["tfidf"] = tfidf

    # ── Baseline ANN ────────────────────────────────────────────────────────
    baseline_weights = _load_latest("ANN_Baseline", MODELS_DIR_PATH, ext="pt")
    if baseline_weights and tfidf:
        input_dim = len(tfidf.get_feature_names_out())
        baseline_model = BaselineANN(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
        baseline_model.load_state_dict(baseline_weights)
        baseline_model.eval()
        state["baseline_model"] = baseline_model
    else:
        state["baseline_model"] = None

    # ── Pro ANN ─────────────────────────────────────────────────────────────
    pro_weights = _load_latest("ANN_Pro", MODELS_DIR_PATH, ext="pt")
    if pro_weights and le_pro:
        pro_model = ProANN(input_dim=768, num_classes=num_classes).to(DEVICE)
        pro_model.load_state_dict(pro_weights)
        pro_model.eval()
        state["pro_model"] = pro_model
    else:
        state["pro_model"] = None

    # ── Regression ──────────────────────────────────────────────────────────
    state["regressor"] = _load_latest("Ridge_Regression", MODELS_DIR_PATH)

    # ── KNN ─────────────────────────────────────────────────────────────────
    knn = _load_latest("KNN_Recommender", MODELS_DIR_PATH)
    state["knn"] = knn

    # ── Corpus for recommendations ──────────────────────────────────────────
    import pandas as pd
    corpus_path = PROCESSED_DIR / "corpus.csv"
    if corpus_path.exists():
        state["corpus_df"] = pd.read_csv(corpus_path)
        logger.info("Corpus loaded: %d rows", len(state["corpus_df"]))
    else:
        state["corpus_df"] = None

    # ── TF-IDF corpus matrix ─────────────────────────────────────────────────
    if tfidf is not None and state["corpus_df"] is not None:
        X_corpus = tfidf.transform(
            state["corpus_df"]["lemmas"].fillna("").values
        ).toarray().astype(np.float32)
        state["X_corpus"] = X_corpus
    else:
        state["X_corpus"] = None

    logger.info("Model loading complete.")
    MODEL_STATE.update(state)


# ---------------------------------------------------------------------------
# Lifespan — load models on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting — loading models …")
    try:
        load_all_models()
        logger.info("✓ Models ready.")
    except Exception as exc:
        logger.warning("Model loading skipped (not yet trained): %s", exc)
    yield
    logger.info("API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Philosophical Text Engine API",
    description=(
        "Production ML system for philosophical text analysis. "
        "Classifies philosophical schools, scores reading complexity, "
        "and recommends similar readings."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str
    model_tier: Optional[str] = "baseline"  # "baseline" or "pro"


class PredictionResponse(BaseModel):
    predicted_school:   str
    confidence_score:   float
    complexity_index:   float
    top_3_recommendations: list
    model_used:         str
    all_probabilities:  Optional[dict] = None


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------
def run_inference(text: str, model_tier: str = "baseline") -> dict:
    """
    Run the full prediction pipeline on an input text.

    Args:
        text:       Raw philosophical text snippet.
        model_tier: "baseline" (TF-IDF + 1-layer ANN) or "pro" (DistilBERT).

    Returns:
        Dict matching PredictionResponse schema.
    """
    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=422, detail="Text is too short (min 10 chars).")

    # ── Step 1: Complexity score ─────────────────────────────────────────────
    complexity = compute_complexity_score(text)

    # ── Step 2: Classification ───────────────────────────────────────────────
    predicted_school = "Unknown"
    confidence_score = 0.0
    all_probs = {}

    if model_tier == "pro" and MODEL_STATE.get("pro_model"):
        model     = MODEL_STATE["pro_model"]
        le        = MODEL_STATE["le_pro"]
        embedding = get_distilbert_embeddings([text[:512]])
        tensor    = torch.tensor(embedding, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        pred_idx         = int(np.argmax(probs))
        predicted_school = le.classes_[pred_idx]
        confidence_score = round(float(probs[pred_idx]), 4)
        all_probs        = {c: round(float(p), 4) for c, p in zip(le.classes_, probs)}
        model_used       = "Pro (DistilBERT + Deep ANN)"

    elif MODEL_STATE.get("baseline_model") and MODEL_STATE.get("tfidf"):
        model   = MODEL_STATE["baseline_model"]
        tfidf   = MODEL_STATE["tfidf"]
        le      = MODEL_STATE["le_baseline"]
        vec     = tfidf.transform([text]).toarray().astype(np.float32)
        tensor  = torch.tensor(vec, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        pred_idx         = int(np.argmax(probs))
        predicted_school = le.classes_[pred_idx]
        confidence_score = round(float(probs[pred_idx]), 4)
        all_probs        = {c: round(float(p), 4) for c, p in zip(le.classes_, probs)}
        model_used       = "Baseline (TF-IDF + 1-layer ANN)"

    else:
        model_used = "No model loaded — run the Prefect pipeline first."

    # ── Step 3: Recommendations ──────────────────────────────────────────────
    recommendations = []
    if (MODEL_STATE.get("knn") and MODEL_STATE.get("X_corpus") is not None
            and MODEL_STATE.get("corpus_df") is not None):
        tfidf = MODEL_STATE.get("tfidf")
        if tfidf:
            q_vec = tfidf.transform([text]).toarray().astype(np.float32)[0]
            knn_artifacts = {
                "model": MODEL_STATE["knn"],
                "df":    MODEL_STATE["corpus_df"],
                "X":     MODEL_STATE["X_corpus"],
            }
            recommendations = get_recommendations(q_vec, knn_artifacts)

    return {
        "predicted_school":    predicted_school,
        "confidence_score":    confidence_score,
        "complexity_index":    complexity,
        "top_3_recommendations": recommendations,
        "model_used":          model_used,
        "all_probabilities":   all_probs,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    """Health check and API overview."""
    return {
        "status":      "online",
        "service":     "Philosophical Text Engine API",
        "version":     "1.0.0",
        "endpoints":   ["/predict", "/predict/upload", "/schools", "/metrics", "/docs"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_json(request: PredictRequest):
    """
    Predict philosophical school, complexity, and recommendations from raw JSON text.

    - **text**: Philosophical text snippet (min 10 characters).
    - **model_tier**: `"baseline"` (fast) or `"pro"` (accurate, slower).
    """
    return run_inference(text=request.text, model_tier=request.model_tier)


@app.post("/predict/upload", response_model=PredictionResponse, tags=["Inference"])
async def predict_file(
    file: UploadFile = File(..., description="A .txt file containing philosophical text"),
    model_tier: str  = Form(default="baseline"),
):
    """
    Predict from an uploaded .txt file.

    - **file**: A .txt file containing philosophical text.
    - **model_tier**: `"baseline"` or `"pro"`.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=415, detail="Only .txt files are accepted.")
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    return run_inference(text=text, model_tier=model_tier)


@app.get("/schools", tags=["Info"])
def list_schools():
    """Return all philosophical schools the system can classify."""
    schools = [
        {"school": "Stoicism",           "examples": ["Marcus Aurelius", "Epictetus"]},
        {"school": "Existentialism",     "examples": ["Nietzsche", "Kierkegaard"]},
        {"school": "Islamic Philosophy", "examples": ["Al-Ghazali"]},
        {"school": "Rationalism",        "examples": ["Descartes", "Spinoza"]},
        {"school": "Empiricism",         "examples": ["John Locke", "David Hume"]},
        {"school": "Idealism",           "examples": ["Plato", "Kant"]},
    ]
    return {"schools": schools, "total": len(schools)}


@app.get("/metrics", tags=["Info"])
def get_metrics():
    """Return metrics from the last successful Prefect pipeline run."""
    results_path = PROCESSED_DIR / "pipeline_results.json"
    if not results_path.exists():
        return {"message": "No pipeline results found. Run the Prefect pipeline first."}
    return json.loads(results_path.read_text())


@app.post("/reload-models", tags=["Admin"])
def reload_models():
    """Reload all model artifacts from disk (after a new pipeline run)."""
    try:
        load_all_models()
        return {"status": "Models reloaded successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
