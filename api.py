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


class MetaResponse(BaseModel):
    cluster_centers_2d: list[list[float]]
    cluster_names: list[str]        # Majority-vote Tier-2 label per cluster
    cluster_colors: list[str]       # Consistent hex color per school


# Consistent color palette for each philosophical school
_SCHOOL_COLORS: dict[str, str] = {
    "Idealism":       "#7c6ef5",
    "Materialism":    "#34d399",
    "Rationalism":    "#60a5fa",
    "Empiricism":     "#f59e0b",
    "Existentialism": "#f87171",
    "Nihilism":       "#a78bfa",
    "Stoicism":       "#6ee7b7",
}
_DEFAULT_COLOR = "#94a3b8"


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
    Returns K-Means cluster metadata for visualization:
    - Cluster centroids projected into 2D PCA space
    - Majority-vote Tier-2 label compared against actual training labels
    - Consistent color per philosophical school
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded.")
    try:
        import numpy as np
        from collections import Counter
        from taxonomy import IDX_TO_TIER2
        from ingestion import ingest_all
        from preprocessing import clean_and_lemmatize

        centers_10k = engine.kmeans.cluster_centers_
        centers_2d  = engine.pca.transform(centers_10k).tolist()
        n_clusters  = len(centers_2d)

        # ── Majority-vote labelling ──────────────────────────────────────────
        # Load the actual training texts, predict cluster membership, compare
        # against true Tier-2 labels → the winning school labels the cluster.
        try:
            raw_texts, _, y_t2 = ingest_all()
            clean_texts = [clean_and_lemmatize(t) for t in raw_texts]
            X = engine.tfidf_vec.transform(clean_texts)
            cluster_assignments = engine.kmeans.predict(X)

            buckets: dict[int, list[int]] = {i: [] for i in range(n_clusters)}
            for sample_cluster, true_label_idx in zip(cluster_assignments, y_t2):
                buckets[int(sample_cluster)].append(int(true_label_idx))

            cluster_names = [
                IDX_TO_TIER2[Counter(buckets[i]).most_common(1)[0][0]]
                if buckets[i] else "Unknown"
                for i in range(n_clusters)
            ]
        except Exception:
            # Fast fallback: run centroids through the neural network
            import torch
            from engine_core import predict_nn
            cluster_names = []
            for c in centers_10k:
                x = torch.tensor(c, dtype=torch.float32).unsqueeze(0).to(engine.device)
                _, p2 = predict_nn(engine.mlp, x)
                cluster_names.append(IDX_TO_TIER2[int(np.argmax(p2[0]))])

        cluster_colors = [
            _SCHOOL_COLORS.get(name, _DEFAULT_COLOR) for name in cluster_names
        ]

        return MetaResponse(
            cluster_centers_2d=centers_2d,
            cluster_names=cluster_names,
            cluster_colors=cluster_colors,
        )
    except Exception as exc:
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
