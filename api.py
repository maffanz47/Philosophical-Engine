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
    speed: str = Field(
        default="slow",
        description="Execution speed: 'slow' (full model) or 'fast' (25% data model)."
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


class MetaResponse(BaseModel):
    cluster_centers_2d: list[list[float]]
    cluster_names:      list[str]   # Tier-2 school labels
    cluster_tier1:      list[str]   # Tier-1 branch labels


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


@app.get("/meta", response_model=MetaResponse, tags=["meta"])
def get_meta():
    """
    Returns K-Means cluster centroids projected into 2D PCA space,
    labeled with the Tier-2 school and Tier-1 branch that each
    cluster dominantly represents (determined by SVM majority vote).
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded.")
    try:
        import numpy as np
        from taxonomy import IDX_TO_TIER1, IDX_TO_TIER2, TIER2_TO_TIER1

        centers_10k = engine.kmeans_slow.cluster_centers_          # shape (7, 10000)
        centers_2d  = engine.pca.transform(centers_10k)       # shape (7, 2)

        # ── Label each cluster using SVM prediction on the centroid ──────────
        # SVM was trained on Tier-1, so we get the branch.
        # For Tier-2 we use the MLP's argmax on the centroid vector.
        import torch
        from engine_core import predict_nn

        cluster_names = []   # Tier-2
        cluster_tier1 = []   # Tier-1

        for c in centers_10k:
            # --- Tier-1 via SVM on the centroid (sparse-safe) ---
            import scipy.sparse as sp
            c_sparse = sp.csr_matrix(c.reshape(1, -1))
            t1_idx   = int(engine.svm_slow.predict(c_sparse)[0])
            t1_label = IDX_TO_TIER1[t1_idx]

            # --- Tier-2 via MLP on the centroid ---
            c_tensor = torch.tensor(c, dtype=torch.float32).unsqueeze(0).to(engine.device)
            _, p2    = predict_nn(engine.mlp, c_tensor)
            t2_idx   = int(p2[0].argmax())
            t2_label = IDX_TO_TIER2[t2_idx]

            cluster_names.append(t2_label)
            cluster_tier1.append(t1_label)

        return MetaResponse(
            cluster_centers_2d=centers_2d.tolist(),
            cluster_names=cluster_names,
            cluster_tier1=cluster_tier1,
        )
    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


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
        raw = engine.predict(body.text, mode=body.speed)
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

from fastapi.staticfiles import StaticFiles

# Serve the static frontend if the directory exists
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="frontend")
elif os.path.isdir("docs"):
    app.mount("/", StaticFiles(directory="docs", html=True), name="frontend")
