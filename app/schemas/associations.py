from pydantic import BaseModel


class AssociationRule(BaseModel):
    antecedent: list[str]
    consequent: list[str]
    support: float
    confidence: float
    lift: float


class AssociationsResponse(BaseModel):
    rules: list[AssociationRule]
    concept_filter: str | None = None
