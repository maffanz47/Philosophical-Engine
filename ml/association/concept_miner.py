from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssociationRule:
    antecedent: list[str]
    consequent: list[str]
    support: float
    confidence: float
    lift: float


def mine_associations(concept: str) -> list[dict[str, Any]]:
    """
    Association rules mining (Phase 1 placeholder).

    Phase 2+ will:
      - build transactions from each book's top_concepts
      - run Apriori (mlxtend) with min_support=0.05 and min_confidence=0.3
      - mine author co-occurrence patterns
      - filter rules that contain `concept` in antecedent or consequent
    """
    _ = concept
    return []
