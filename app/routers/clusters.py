from fastapi import APIRouter
from pydantic import BaseModel

from ml.clustering.thought_clusterer import get_clusters as get_clusters_stub

router = APIRouter(prefix="/clusters", tags=["clustering"])


class ClusterInfoItem(BaseModel):
    cluster_id: int
    label: str
    size: int
    top_terms: list[str]
    representative_books: list[str]


@router.get("/", response_model=list[ClusterInfoItem])
def clusters() -> list[ClusterInfoItem]:
    """
    Phase 1:
      - Returns cluster metadata when artifacts exist (currently stub returns []).
    """
    results = get_clusters_stub()

    items: list[ClusterInfoItem] = []
    for r in results:
        if isinstance(r, dict):
            items.append(ClusterInfoItem(**r))
    return items
