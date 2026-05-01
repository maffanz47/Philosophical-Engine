"""Prefect orchestration flow for ingestion, preprocessing, and training."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from prefect import flow, get_run_logger, task
from prefect.exceptions import MissingContextError

from src.data_loader import IngestionConfig, ingest_gutenberg_book
from src.feature_engineering import distilbert_features, tfidf_features
from src.models import train_and_evaluate_tier
from src.preprocessor import preprocess_text
from src.recommender import project_pca_2d, run_kmeans


def _safe_log(message: str, *args: object) -> None:
    """Log with Prefect logger when available, otherwise print."""
    try:
        get_run_logger().info(message, *args)
    except MissingContextError:
        print(message % args if args else message)


@dataclass
class FlowConfig:
    """Configuration for the philosophical engine flow."""

    book_ids: List[int]
    model_output_dir: str = "artifacts/models"
    reports_output_dir: str = "artifacts/reports"
    discord_webhook_url: Optional[str] = None
    num_classes: int = 6
    tfidf_max_features: int = 1024


@task(retries=2, retry_delay_seconds=5)
def ingest_task(book_id: int) -> List[str]:
    """Fetch and chunk one Gutenberg title."""
    _safe_log("Ingesting book_id=%s", book_id)
    return ingest_gutenberg_book(book_id, IngestionConfig())


@task
def preprocessing_task(chunks_by_book: Dict[int, List[str]]) -> Dict[int, List[List[str]]]:
    """Tokenize and lemmatize all chunks grouped by book."""
    processed: Dict[int, List[List[str]]] = {}

    for book_id, chunks in chunks_by_book.items():
        processed[book_id] = [preprocess_text(chunk) for chunk in chunks]
        _safe_log("Preprocessed %s chunks for book_id=%s", len(processed[book_id]), book_id)

    return processed


@task
def training_and_evaluation_task(processed_corpus: Dict[int, List[List[str]]], config: FlowConfig) -> Dict[str, Dict[str, object]]:
    """
    Train baseline and pro models concurrently and expose required metrics.

    Fast tier: TF-IDF + shallow ANN.
    Pro tier: DistilBERT embeddings + deep ANN.
    """
    dataset_texts: List[str] = []
    labels: List[int] = []
    complexities: List[float] = []
    for book_id, tokenized_chunks in processed_corpus.items():
        class_id = int(book_id % config.num_classes)
        for tokens in tokenized_chunks:
            if not tokens:
                continue
            text = " ".join(tokens)
            dataset_texts.append(text)
            labels.append(class_id)
            complexity = float(len(tokens) / max(1, len(set(tokens))))
            complexities.append(complexity)

    if len(dataset_texts) < 4:
        raise ValueError("Insufficient training samples. Add more books/chunks for model training.")

    y = np.array(labels, dtype=int)
    c = np.array(complexities, dtype=np.float32)
    x_fast, _ = tfidf_features(dataset_texts, max_features=config.tfidf_max_features)
    x_pro = distilbert_features(dataset_texts)

    with ThreadPoolExecutor(max_workers=2) as executor:
        baseline_future = executor.submit(
            train_and_evaluate_tier, "Fast", x_fast, y, c, config.model_output_dir, config.num_classes
        )
        pro_future = executor.submit(
            train_and_evaluate_tier, "Pro", x_pro, y, c, config.model_output_dir, config.num_classes
        )

        baseline_artifact = baseline_future.result()
        pro_artifact = pro_future.result()

    _safe_log(
        "Training complete. fast_f1=%s pro_f1=%s fast_rmse=%s pro_rmse=%s",
        baseline_artifact["f1_score"],
        pro_artifact["f1_score"],
        baseline_artifact["rmse"],
        pro_artifact["rmse"],
    )
    return {
        "baseline": baseline_artifact,
        "pro": pro_artifact,
    }


@task
def reporting_task(processed_corpus: Dict[int, List[List[str]]], training_artifacts: Dict[str, Dict[str, object]], config: FlowConfig) -> Dict[str, object]:
    """Persist metrics and unsupervised analysis outputs for reporting."""
    report_dir = Path(config.reports_output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    doc_vectors: List[np.ndarray] = []
    doc_index: List[Dict[str, int]] = []
    for book_id, tokenized_chunks in processed_corpus.items():
        for chunk_idx, tokens in enumerate(tokenized_chunks):
            token_count = len(tokens)
            unique_count = len(set(tokens))
            avg_len = float(sum(len(token) for token in tokens) / max(1, token_count))
            doc_vectors.append(np.array([token_count, unique_count, avg_len], dtype=float))
            doc_index.append({"book_id": book_id, "chunk_idx": chunk_idx})

    if not doc_vectors:
        payload = {"message": "No chunks available for report generation."}
        metrics_path = report_dir / "metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"metrics_path": str(metrics_path), "cluster_path": None, "pca_path": None}

    feature_matrix = np.vstack(doc_vectors)
    unique_rows = np.unique(feature_matrix, axis=0)
    n_clusters = min(5, len(feature_matrix), len(unique_rows))
    cluster_assignments = run_kmeans(feature_matrix, n_clusters=n_clusters)
    pca_projection = project_pca_2d(feature_matrix)

    metrics_payload = {
        "training": training_artifacts,
        "documents_analyzed": len(doc_vectors),
        "feature_dimensions": int(feature_matrix.shape[1]),
        "n_clusters": int(n_clusters),
    }
    cluster_payload = [
        {**meta, "cluster": int(cluster_assignments[idx])}
        for idx, meta in enumerate(doc_index)
    ]
    pca_payload = [
        {**meta, "x": float(pca_projection[idx][0]), "y": float(pca_projection[idx][1])}
        for idx, meta in enumerate(doc_index)
    ]

    metrics_path = report_dir / "metrics.json"
    clusters_path = report_dir / "clusters.json"
    pca_path = report_dir / "pca_2d.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    clusters_path.write_text(json.dumps(cluster_payload, indent=2), encoding="utf-8")
    pca_path.write_text(json.dumps(pca_payload, indent=2), encoding="utf-8")

    _safe_log("Wrote report artifacts to %s", report_dir)
    return {"metrics_path": str(metrics_path), "cluster_path": str(clusters_path), "pca_path": str(pca_path)}


@task
def notify_discord_task(webhook_url: Optional[str], payload: Dict[str, object]) -> None:
    """Send success/failure updates to Discord webhook if configured."""
    if not webhook_url:
        _safe_log("No Discord webhook configured; skipping notification.")
        return

    message = {"content": f"Philosophical Engine flow update:\n```{payload}```"}
    response = requests.post(webhook_url, json=message, timeout=10)
    response.raise_for_status()
    _safe_log("Discord notification sent.")


@flow(name="philosophical-engine-main-flow")
def run_main_flow(config: FlowConfig) -> Dict[str, object]:
    """Run the full end-to-end ETL and model orchestration flow."""
    _safe_log("Starting flow with config=%s", asdict(config))

    chunks_by_book: Dict[int, List[str]] = {}
    for book_id in config.book_ids:
        chunks_by_book[book_id] = ingest_task(book_id)

    processed_corpus = preprocessing_task(chunks_by_book)
    training_artifacts = training_and_evaluation_task(processed_corpus, config)
    report_artifacts = reporting_task(processed_corpus, training_artifacts, config)

    result = {
        "books_ingested": len(chunks_by_book),
        "processed_books": len(processed_corpus),
        "training": training_artifacts,
        "reports": report_artifacts,
    }

    notify_discord_task(config.discord_webhook_url, result)
    return result


if __name__ == "__main__":
    sample_config = FlowConfig(book_ids=[1342, 2701], discord_webhook_url=None)
    run_main_flow(sample_config)
