from fastapi import APIRouter
from pydantic import BaseModel

from ml.association.concept_miner import mine_associations as mine_associations_stub

router = APIRouter(prefix="/associations", tags=["associations"])


class AssociationRuleItem(BaseModel):
    antecedent: list[str]
    consequent: list[str]
    support: float
    confidence: float
    lift: float


@router.get("/", response_model=list[AssociationRuleItem])
def associations(concept: str) -> list[AssociationRuleItem]:
    """
    Phase 1:
      - Returns association rules when artifacts exist (currently stub returns []).
    """
    results = mine_associations_stub(concept)

    items: list[AssociationRuleItem] = []
    for r in results:
        if isinstance(r, dict):
            items.append(AssociationRuleItem(**r))
    return items
