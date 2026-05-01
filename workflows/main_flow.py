"""
main_flow.py
============
Prefect 3.0 orchestration pipeline for the Philosophical Text Engine.

Flow stages:
  1. ingestion_task      — Download all Gutenberg books
  2. preprocessing_task  — Tokenize, lemmatize, chunk, score complexity
  3. training_task       — Train Baseline & Pro classifiers + Regression (concurrent)
  4. clustering_task     — KMeans clustering + PCA/t-SNE visualization
  5. knn_task            — Build KNN recommender index
  6. notification_task   — Send Discord webhook notification
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import httpx
from dotenv import load_dotenv

# Load env
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_PATH)

DISCORD_WEBHOOK_URL = os.getenv("discord_webhook_url", "")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main_flow")

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
import sys
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'src')))

from data_loader import ingest_all_books
from preprocessor import build_corpus
from models import (
    train_baseline_classifier,
    train_pro_classifier,
    train_regression_model,
    build_knn_recommender,
    run_kmeans_clustering,
    reduce_dimensions,
)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Tasks
# ===========================================================================

@task(
    name="ingestion-task",
    retries=3,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=1),
    log_prints=True,
)
def ingestion_task(force_download: bool = False) -> list:
    """
    Download all philosophical texts from Project Gutenberg.

    Args:
        force_download: Re-download even if local copies exist.

    Returns:
        Manifest list of downloaded book metadata.
    """
    logger.info("Starting ingestion task …")
    manifest = ingest_all_books(force=force_download)
    logger.info("Ingestion complete: %d books.", len(manifest))
    return manifest


@task(
    name="preprocessing-task",
    retries=2,
    retry_delay_seconds=5,
    log_prints=True,
)
def preprocessing_task(manifest: list):
    """
    Tokenize, lemmatize, chunk, and score all downloaded books.

    Args:
        manifest: Output from ingestion_task.

    Returns:
        Processed corpus as a pandas DataFrame.
    """
    import pandas as pd
    logger.info("Starting preprocessing task …")

    # Check for cached corpus
    corpus_path = PROCESSED_DIR / "corpus.csv"
    if corpus_path.exists():
        logger.info("Loading cached corpus from %s", corpus_path)
        df = pd.read_csv(corpus_path)
    else:
        df = build_corpus(manifest, save_csv=True)

    logger.info("Preprocessing complete: %d chunks.", len(df))
    return df


@task(name="baseline-training-task", log_prints=True)
def baseline_training_task(df):
    """Train the Tier A Baseline classifier (TF-IDF + 1-layer ANN)."""
    logger.info("Training Baseline model …")
    artifacts = train_baseline_classifier(df)
    logger.info("Baseline F1: %.4f", artifacts["f1_score"])
    return artifacts


@task(name="pro-training-task", log_prints=True)
def pro_training_task(df):
    """Train the Tier B Pro classifier (DistilBERT + Deep ANN)."""
    logger.info("Training Pro model …")
    artifacts = train_pro_classifier(df)
    logger.info("Pro F1: %.4f", artifacts["f1_score"])
    return artifacts


@task(name="regression-training-task", log_prints=True)
def regression_training_task(df, embeddings):
    """Train Ridge Regression for complexity score prediction."""
    logger.info("Training Regression model …")
    artifacts = train_regression_model(df, embeddings=embeddings)
    logger.info("Regression RMSE: %.4f", artifacts["rmse"])
    return artifacts


@task(name="clustering-task", log_prints=True)
def clustering_task(df, X):
    """Run KMeans clustering and generate PCA/t-SNE visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("Running clustering …")
    df_clustered = run_kmeans_clustering(X, df, n_clusters=6)

    # PCA visualization
    coords_pca = reduce_dimensions(X, method="pca")
    fig, ax = plt.subplots(figsize=(10, 7))
    schools = df["school"].unique()
    colors = plt.cm.tab10(range(len(schools)))
    for school, color in zip(schools, colors):
        mask = df["school"] == school
        ax.scatter(coords_pca[mask, 0], coords_pca[mask, 1],
                   label=school, color=color, alpha=0.6, s=15)
    ax.set_title("PCA — Philosophical Text Embeddings")
    ax.legend(loc="best", fontsize=8)
    pca_path = PROCESSED_DIR / "pca_visualization.png"
    fig.savefig(pca_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PCA plot saved → %s", pca_path)

    return df_clustered


@task(name="knn-task", log_prints=True)
def knn_task(X_pro, X_baseline, df):
    """Build the KNN recommendation indices."""
    logger.info("Building KNN recommender (Baseline) …")
    baseline_artifacts = build_knn_recommender(X_baseline, df, n_neighbors=4, prefix="KNN_Baseline")
    
    logger.info("Building KNN recommender (Pro) …")
    pro_artifacts = build_knn_recommender(X_pro, df, n_neighbors=4, prefix="KNN_Pro")
    
    return baseline_artifacts, pro_artifacts


@task(name="notification-task", retries=2, log_prints=True)
def notification_task(status: str, metrics: dict):
    """
    Send a success/failure notification to Discord webhook.

    Args:
        status:  "SUCCESS" or "FAILURE"
        metrics: Dict of key metrics to include in the message.
    """
    if not DISCORD_WEBHOOK_URL:
        logger.warning("No DISCORD_WEBHOOK_URL set — skipping notification.")
        return

    color = 0x00FF00 if status == "SUCCESS" else 0xFF0000
    timestamp = datetime.utcnow().isoformat() + "Z"

    fields = [
        {"name": k, "value": str(v), "inline": True}
        for k, v in metrics.items()
    ]

    payload = {
        "embeds": [{
            "title": f"🧠 Philosophical Text Engine — Pipeline {status}",
            "color": color,
            "timestamp": timestamp,
            "fields": fields,
            "footer": {"text": "Prefect 3.0 Orchestration"},
        }]
    }

    try:
        response = httpx.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Discord notification sent (status=%s).", status)
    except Exception as exc:
        logger.error("Discord notification failed: %s", exc)


# ===========================================================================
# Main Prefect Flow
# ===========================================================================
@flow(name="philosophical-engine-pipeline", log_prints=True)
def philosophical_engine_pipeline(force_download: bool = False):
    """
    Full Prefect 3.0 pipeline:
      ingestion → preprocessing → [baseline | pro | regression] → clustering → knn → notify
    """
    import numpy as np
    metrics = {}

    try:
        # Stage 1: Ingestion
        manifest = ingestion_task(force_download=force_download)
        metrics["books_downloaded"] = len(manifest)

        # Stage 2: Preprocessing
        df = preprocessing_task(manifest)
        metrics["total_chunks"] = len(df)

        # Pro model training
        pro_artifacts = pro_training_task(df)
        metrics["pro_f1"] = round(pro_artifacts["f1_score"], 4)
        
        # Validation: check for F1 < 0.70 for any school
        low_f1_schools = {k: v for k, v in pro_artifacts["class_f1_scores"].items() if v < 0.70}
        if low_f1_schools:
            err_msg = f"Validation failed: F1-score below 0.70 for schools: {low_f1_schools}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        X_embeddings = pro_artifacts["X_embeddings"]

        # Stage 3: Concurrent training of Baseline + Regression
        baseline_future = baseline_training_task.submit(df)
        regression_future = regression_training_task.submit(df, X_embeddings)

        baseline_artifacts = baseline_future.result()
        regression_artifacts = regression_future.result()

        metrics["baseline_f1"] = round(baseline_artifacts["f1_score"], 4)
        metrics["regression_rmse"] = round(regression_artifacts["rmse"], 4)

        # Stage 4: Clustering (on embeddings)
        df_clustered = clustering_task(df, X_embeddings)

        # Stage 5: KNN Recommender
        tfidf = baseline_artifacts["tfidf"]
        X_baseline = tfidf.transform(df["lemmas"].fillna("").values).toarray().astype(np.float32)
        knn_baseline_artifacts, knn_pro_artifacts = knn_task(X_embeddings, X_baseline, df)

        # Save artifacts manifest for FastAPI to load
        artifacts_manifest = {
            "baseline_f1":      metrics["baseline_f1"],
            "pro_f1":           metrics["pro_f1"],
            "regression_rmse":  metrics["regression_rmse"],
            "total_chunks":     metrics["total_chunks"],
            "schools":          df["school"].value_counts().to_dict(),
            "pipeline_run_at":  datetime.utcnow().isoformat() + "Z",
        }
        manifest_out = Path(__file__).resolve().parent.parent / "data" / "processed" / "pipeline_results.json"
        manifest_out.write_text(json.dumps(artifacts_manifest, indent=2))
        logger.info("Pipeline results saved → %s", manifest_out)

        # Stage 6: Notify success
        notification_task(status="SUCCESS", metrics=metrics)

        logger.info("✓ Pipeline complete! Metrics: %s", metrics)
        return {"status": "SUCCESS", "metrics": metrics}

    except Exception as exc:
        logger.error("Pipeline FAILED: %s", exc, exc_info=True)
        notification_task(status="FAILURE", metrics={"error": str(exc)})
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv(ENV_PATH)
    result = philosophical_engine_pipeline(force_download=False)
    print(result)
