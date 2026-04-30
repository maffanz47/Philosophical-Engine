from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class InfluencePredictResult:
    influence_score: float


def _default_result() -> InfluencePredictResult:
    return InfluencePredictResult(influence_score=0.0)


def _latest_model_path(models_dir: Path) -> Path | None:
    if not models_dir.exists():
        return None
    candidates = sorted(models_dir.glob("influence_predictor_v*.pkl"), reverse=True)
    return candidates[0] if candidates else None


def predict_influence(features: dict[str, Any]) -> float:
    """
    Predict influence_score in [0, 1].

    Phase 1 behavior:
      - If no trained artifact exists, returns 0.0.
      - In Phase 2+, this will use structured + TF-IDF features and a regressor,
        then apply scaling to [0, 1].
    """
    models_dir = Path("models/regression")
    model_path = _latest_model_path(models_dir)
    if model_path is None:
        return _default_result().influence_score

    try:
        artifact = joblib.load(model_path)
    except Exception:
        return _default_result().influence_score

    model = artifact.get("model")
    if model is None:
        return _default_result().influence_score

    # Phase 2+ will build the actual feature vector using TF-IDF + structured features.
    # For now, we cannot reliably construct the vector, so return a bounded fallback.
    try:
        raw_pred = float(model.predict([[]])[0])  # type: ignore[attr-defined]
    except Exception:
        return _default_result().influence_score

    # Heuristic clamp to [0, 1]
    if raw_pred != raw_pred:  # NaN
        return _default_result().influence_score
    return max(0.0, min(1.0, raw_pred))
