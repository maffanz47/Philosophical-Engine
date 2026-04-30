from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel

from ml.dimensionality.embedding_reducer import get_embeddings_map

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class EmbeddingsMapItem(BaseModel):
    id: str
    title: str
    x: float
    y: float
    school: str
    era: str


@router.get("/map", response_model=list[EmbeddingsMapItem])
def embeddings_map() -> list[EmbeddingsMapItem]:
    """
    Returns the 2D UMAP map for frontend scatter plotting.

    Phase 1: returns [] because embeddings artifacts are not generated yet.
    """
    items = get_embeddings_map()
    return [EmbeddingsMapItem(**asdict(item)) for item in items]
