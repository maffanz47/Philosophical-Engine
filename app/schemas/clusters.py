from pydantic import BaseModel


class ClusterItem(BaseModel):
    cluster_id: int
    label: str
    size: int
    top_terms: list[str]
    representative_books: list[str]


class ClustersResponse(BaseModel):
    clusters: list[ClusterItem]
