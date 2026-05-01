from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from ml.dimensionality.embedding_reducer import get_embeddings_map

router = APIRouter(prefix="/embeddings", tags=["Dimensionality Reduction"])

@router.get("/map", response_model=List[Dict[str, Any]])
async def get_map():
    """Returns a JSON array of 2D UMAP coordinates with metadata."""
    try:
        data = get_embeddings_map()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
