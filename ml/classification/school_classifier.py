from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


SCHOOLS: list[str] = [
    "Empiricism",
    "Rationalism",
    "Existentialism",
    "Stoicism",
    "Idealism",
    "Pragmatism",
    "Other",
]


@dataclass(frozen=True)
class SchoolPredictResult:
    school: str
    confidence: float
    top3: list[str]


def _default_result() -> SchoolPredictResult:
    return SchoolPredictResult(school="Other", confidence=0.0, top3=["Other"])


def _try_load_artifact(artifact_path: Path) -> Any | None:
    if not artifact_path.exists():
        return None
    try:
        return joblib.load(artifact_path)
    except Exception:
        return None


def _latest_model_path(models_dir: Path) -> Path | None:
    if not models_dir.exists():
        return None
    candidates = sorted(models_dir.glob("school_classifier_v*.pkl"), reverse=True)
    return candidates[0] if candidates else None


def predict_school(text: str) -> dict[str, Any]:
    """
    Predict philosophical school for an input text.

    Phase 1 behavior:
      - If no trained artifact exists, returns a safe default.
      - In later phases, this will load TF-IDF + structured-feature models
        and (optionally) a fine-tuned DistilBERT classifier.

    Args:
        text: Free-form input text.

    Returns:
        dict with {school, confidence, top3}.
    """
    models_dir = Path("models/classification")
    model_path = _latest_model_path(models_dir)

    if model_path is None:
        return _default_result().__dict__

    # Expecting a joblib dict with keys: "model", "tfidf", "structured_scaler" (later)
    artifact = _try_load_artifact(model_path)
    if artifact is None:
        return _default_result().__dict__

    model = artifact.get("model")
    tfidf = artifact.get("tfidf")

    if model is None or tfidf is None:
        return _default_result().__dict__

    # Structured features will be added in Phase 2+.
    # For now use TF-IDF features only.
    try:
        X = tfidf.transform([text])
        proba = model.predict_proba(X)[0]
        top_idx = proba.argsort()[::-1][:3]
        top3 = [SCHOOLS[int(i)] if int(i) < len(SCHOOLS) else "Other" for i in top_idx]
        best_i = int(top_idx[0])
        best_school = SCHOOLS[best_i] if best_i < len(SCHOOLS) else "Other"
        best_conf = float(proba[best_i])
        return {"school": best_school, "confidence": best_conf, "top3": top3}
    except Exception:
        return _default_result().__dict__


def train_school_classifier(df: "Any") -> None:
    """
    Train school classifiers (Phase 2+ implementation).

    This function will:
      - derive structured features and TF-IDF vectors
      - train Logistic Regression, XGBoost, and DistilBERT
      - select best by F1-macro
      - log metrics and artifacts to MLflow
      - save best model to models/classification/school_classifier_v{timestamp}.pkl

    Args:
        df: Input DataFrame containing at least columns:
            - full_text (str)
            - avg_sentence_length (float)
            - vocab_richness (float)
            - sentiment_polarity (float)
            - named_entity_count (int)
            - top_concepts (list[str]) OR compatible serialized form
            - era_label (str)
            - school_label (str)

    Notes:
        Phase 1 scaffolding only; raise to make it explicit when called.
    """
    raise NotImplementedError("Phase 1: training will be implemented in Phase 2+.")
