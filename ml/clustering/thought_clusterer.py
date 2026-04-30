from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClusterItem:
    cluster_id: int
    label: str
    size: int
    top_terms: list[str]
    representative_books: list[str]


def get_clusters() -> list[dict[str, Any]]:
    """
    Clustering (Phase 1 placeholder).

    Phase 2+ will:
      - cluster UMAP 2D embeddings using K-Means (auto-k) + HDBSCAN
      - derive human-readable "emergent school" labels per cluster
      - compare to ground-truth school_label via Adjusted Rand Index
      - log metrics to MLflow
    """
    return []
