from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel

from ml.recommendation.recommender import RecommendResult, recommend as recommend_stub

router = APIRouter(prefix="/recommend", tags=["recommendation"])


class RecommendItem(BaseModel):
    title: str
    author: str | None = None
    similarity: float
    reason: str | None = None


@router.get("/", response_model=list[RecommendItem])
def recommend(book_id_or_text: str, n: int = 10) -> list[RecommendItem]:
    """
    Phase 1:
      - Uses the latest recommendation stub/artifacts when available.
      - Currently returns [] until Phase 2+ artifacts exist.
    """
    results = recommend_stub(book_id_or_text, n=n)

    # Phase 1 stub returns list[dict]; future versions may return dataclasses.
    items: list[RecommendItem] = []
    for r in results:
        if isinstance(r, dict):
            items.append(RecommendItem(**r))
        elif hasattr(r, "__dataclass_fields__"):  # defensive
            items.append(RecommendItem(**asdict(r)))  # type: ignore[arg-type]
        else:
            # fallback for unexpected shapes
            items.append(RecommendItem(**getattr(r, "__dict__", {})))  # type: ignore[arg-type]

    return items
