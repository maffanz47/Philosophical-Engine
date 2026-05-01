"""Evaluate classification and regression models with confusion matrices."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import IngestionConfig, ingest_gutenberg_book
from src.feature_engineering import distilbert_features, tfidf_features
from src.models import (
    BaselineClassifier,
    ComplexityRegressor,
    ProClassifier,
    _fit_classifier,
    _fit_regressor,
)


def _synthetic_corpus(num_classes: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
    base_texts = [
        "Virtue, discipline, and rational self-control produce tranquility and resilience.",
        "Freedom and anxiety define existence, and personal choice creates meaning.",
        "Reason and revelation cooperate in the pursuit of metaphysical truth.",
        "Observation and sensory evidence ground practical knowledge.",
    ]
    texts: List[str] = []
    labels: List[int] = []
    complexities: List[float] = []
    for class_id in range(num_classes):
        seed_text = base_texts[class_id % len(base_texts)]
        for repetition in range(8):
            sample = f"{seed_text} Example {repetition} class {class_id}."
            words = sample.split()
            texts.append(sample)
            labels.append(class_id)
            complexities.append(float(len(words) / max(1, len(set(words)))))
    return texts, np.array(labels, dtype=int), np.array(complexities, dtype=np.float32)


def build_dataset(book_ids: List[int], num_classes: int, max_samples: int | None = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    texts: List[str] = []
    labels: List[int] = []
    complexities: List[float] = []
    ingestion_cfg = IngestionConfig(chunk_size_words=350, min_chunk_words=300, max_chunk_words=500)

    per_book_cap = None
    if max_samples is not None and len(book_ids) > 0:
        per_book_cap = max(1, max_samples // len(book_ids))

    for book_id in book_ids:
        try:
            chunks = ingest_gutenberg_book(book_id, ingestion_cfg)
        except Exception:
            chunks = []
        class_id = int(book_id % num_classes)
        added_for_book = 0
        for chunk in chunks:
            words = chunk.split()
            if len(words) < 40:
                continue
            texts.append(chunk)
            labels.append(class_id)
            complexities.append(float(len(words) / max(1, len(set(words)))))
            added_for_book += 1
            if per_book_cap is not None and added_for_book >= per_book_cap:
                break

    if len(texts) < 8:
        return _synthetic_corpus(num_classes)

    return texts, np.array(labels, dtype=int), np.array(complexities, dtype=np.float32)


def hashed_pro_features(texts: List[str], dim: int = 768) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for text in texts:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vectors.append(rng.standard_normal(dim).astype(np.float32))
    return np.vstack(vectors)


def evaluate_tier(
    tier: str,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(
        x, y, c, test_size=0.3, random_state=42, stratify=y
    )
    input_dim = int(x.shape[1])

    if tier == "Fast":
        classifier = BaselineClassifier(input_dim=input_dim, num_classes=num_classes)
    else:
        classifier = ProClassifier(input_dim=input_dim, num_classes=num_classes)
    regressor = ComplexityRegressor(input_dim=input_dim)

    _fit_classifier(classifier, x_train, y_train, epochs=8)
    _fit_regressor(regressor, x_train, c_train, epochs=8)

    classifier.eval()
    regressor.eval()
    with torch.no_grad():
        logits = classifier(torch.tensor(x_test, dtype=torch.float32))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
        c_pred = regressor(torch.tensor(x_test, dtype=torch.float32)).squeeze(1).cpu().numpy()

    cls_metrics = {
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels_present": sorted(np.unique(y_test).tolist()),
    }
    reg_metrics = {
        "rmse": round(float(np.sqrt(mean_squared_error(c_test, c_pred))), 4),
        "mae": round(float(mean_absolute_error(c_test, c_pred)), 4),
        "r2": round(float(r2_score(c_test, c_pred)), 4),
    }
    return {"classification": cls_metrics, "regression": reg_metrics}


def run_evaluation(
    book_ids: List[int],
    num_classes: int,
    output_path: Path,
    use_distilbert: bool = False,
    max_samples: int | None = 120,
) -> Dict[str, object]:
    texts, y, c = build_dataset(book_ids, num_classes, max_samples=max_samples)
    x_fast, _ = tfidf_features(texts, max_features=1024)
    if use_distilbert:
        x_pro = distilbert_features(texts)
    else:
        x_pro = hashed_pro_features(texts)

    fast_result = evaluate_tier("Fast", x_fast, y, c, num_classes)
    pro_result = evaluate_tier("Pro", x_pro, y, c, num_classes)

    report = {
        "book_ids": book_ids,
        "num_samples": len(texts),
        "num_classes": num_classes,
        "pro_feature_source": "distilbert" if use_distilbert else "hashed_fallback",
        "fast_tier": fast_result,
        "pro_tier": pro_result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ANN classifier and regression models.")
    parser.add_argument(
        "--book-ids",
        nargs="+",
        type=int,
        default=[1342, 84, 2701, 11],
        help="Project Gutenberg book ids to build evaluation corpus.",
    )
    parser.add_argument("--num-classes", type=int, default=4, help="Number of synthetic classes.")
    parser.add_argument(
        "--use-distilbert",
        action="store_true",
        help="Use DistilBERT embeddings for pro tier (can be slow).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=120,
        help="Cap total dataset chunks for faster evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/reports/evaluation_report.json",
        help="Path to write evaluation JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_evaluation(
        args.book_ids,
        args.num_classes,
        Path(args.output),
        use_distilbert=args.use_distilbert,
        max_samples=args.max_samples,
    )
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
