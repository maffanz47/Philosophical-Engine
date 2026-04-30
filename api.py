"""
api.py — FastAPI backend for the Philosophical Text Engine.

Loads pre-trained artifacts from engine_artifacts/ and exposes:
  POST /predict  — classify a philosophical text passage
  GET  /health   — readiness probe for Docker/CI

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys, os

# ── Bootstrap: ensure project modules are importable ──────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from engine_core import PhilosophyEngine

# ── Global engine instance (loaded once at startup) ────────────────────────
engine: PhilosophyEngine | None = None

ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "engine_artifacts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models before first request; release on shutdown."""
    global engine
    print(f"[startup] Loading engine artifacts from '{ARTIFACTS_PATH}' …")
    try:
        engine = PhilosophyEngine()
        engine.load(path=ARTIFACTS_PATH)
        print("[startup] Engine ready.")
    except FileNotFoundError as exc:
        print(f"[startup] ERROR: {exc}")
        print("[startup] Run `python train.py` first to generate artifacts.")
        engine = None
    yield
    # Cleanup (if needed) goes here
    engine = None
    print("[shutdown] Engine released.")


# ── Application ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Philosophical Text Engine API",
    description=(
        "Classify philosophical passages using a hybrid TF-IDF + SVM + "
        "PyTorch Neural Network pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow GitHub Pages frontend and local dev to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your GH Pages URL in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        description="A philosophical text passage (minimum 20 characters).",
        examples=["The will to power is the fundamental drive of all existence."],
    )


class Tier1Result(BaseModel):
    label: str
    probabilities: dict[str, float]


class Tier2Result(BaseModel):
    label: str
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    svm_label: str
    nn_tier1: Tier1Result
    nn_tier2: Tier2Result
    kmeans_cluster: int
    pca_x: float
    pca_y: float


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    """Readiness probe — returns 503 if engine failed to load."""
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not loaded. Run train.py first.",
        )
    return {"status": "ok", "engine": "loaded"}


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(body: PredictRequest):
    """
    Classify a philosophical text passage.

    Returns:
    - **svm_label**      — deterministic Tier-1 SVM classification
    - **nn_tier1**       — neural network Tier-1 branch + probabilities
    - **nn_tier2**       — neural network Tier-2 school + probabilities
    - **kmeans_cluster** — unsupervised cluster ID (0–6)
    - **pca_x / pca_y**  — 2D principal component coordinates
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded.")

    try:
        raw = engine.predict(body.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Convert probability string percentages to floats (e.g. "83.4%" → 0.834)
    def parse_probs(d: dict) -> dict[str, float]:
        return {k: float(v.strip("%")) / 100.0 for k, v in d.items()}

    return PredictResponse(
        svm_label=raw["svm_raw"],
        nn_tier1=Tier1Result(
            label=raw["nn_tier1"],
            probabilities=parse_probs(raw["nn_probs_t1"]),
        ),
        nn_tier2=Tier2Result(
            label=raw["nn_tier2"],
            probabilities=parse_probs(raw["nn_probs_t2"]),
        ),
        kmeans_cluster=raw["cluster_id"],
        pca_x=raw["pca_coords"][0],
        pca_y=raw["pca_coords"][1],
    )
