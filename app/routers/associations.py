from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any
from ml.association.concept_miner import get_associations

router = APIRouter(prefix="/associations", tags=["Association Rules"])

@router.get("/", response_model=List[Dict[str, Any]])
async def get_concept_associations(concept: str = Query(..., description="Concept to search rules for")):
    """Returns association rules containing the given concept."""
    try:
        data = get_associations(concept)
        if data and "error" in data[0]:
            raise HTTPException(status_code=500, detail=data[0]["error"])
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
