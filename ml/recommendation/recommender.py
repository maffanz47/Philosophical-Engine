from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecommendResult:
    title: str
    author: str | None
    similarity: float
    reason: str | None


def recommend(book_id_or_text: str, n: int = 10) -> list[dict[str, Any]]:
    """
    Recommend similar works (Phase 1 placeholder).

    Phase 2+ will implement:
      - content-based similarity using sentence embeddings
      - collaborative filtering using implicit feedback proxy
      - hybrid scoring (0.7 content / 0.3 collaborative)
    """
    _ = book_id_or_text
    _ = n
    return []
