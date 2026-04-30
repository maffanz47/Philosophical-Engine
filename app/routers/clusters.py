from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from ml.clustering.thought_clusterer import get_clusters

router = APIRouter(prefix="/clusters", tags=["Clustering"])

@router.get("/", response_model=List[Dict[str, Any]])
async def get_all_clusters():
    """Returns clustered philosophical thoughts."""
    try:
        data = get_clusters()
        if data and "error" in data[0]:
            raise HTTPException(status_code=500, detail=data[0]["error"])
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
