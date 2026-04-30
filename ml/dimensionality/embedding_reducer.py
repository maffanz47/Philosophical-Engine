from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EmbeddingsMapItem:
    id: str
    title: str
    x: float
    y: float
    school: str
    era: str


def get_embeddings_map() -> list[EmbeddingsMapItem]:
    """
    Return the 2D embeddings map for frontend scatter plot.

    Phase 1:
      - If embeddings artifacts don't exist yet, return [].

    Phase 2+:
      - Load embeddings/UMAP coordinates from saved artifacts and return JSON-serializable items.
    """
    embeddings_path = Path("embeddings/umap2d.csv")
    if not embeddings_path.exists():
        return []

    # Placeholder parsing (Phase 2+ will implement robust schema).
    # Keeping Phase 1 intentionally minimal.
    return []


def train_embedding_reducer(df: "Any") -> None:
    """
    Train and generate embeddings + dimensionality reductions (PCA/UMAP/t-SNE).

    Phase 1:
      - Not implemented yet.
    """
    raise NotImplementedError("Phase 1: embedding reducer training will be implemented in Phase 2+.")
