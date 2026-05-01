from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class TrainRequest(BaseModel):
    params: Optional[dict] = None

class BookUploadResponse(BaseModel):
    id: str
    filename: str
    philosopher: str
    status: str

class BookOut(BaseModel):
    id: str
    filename: str
    philosopher: str
    status: str
    page_count: Optional[int]
    chunk_count: int
    uploaded_at: datetime
    processed_at: Optional[datetime]

    class Config:
        from_attributes = True

class ExperimentOut(BaseModel):
    id: str
    module_name: str
    algorithm: str
    accuracy: Optional[float]
    f1_score: Optional[float]
    rmse: Optional[float]
    params_json: dict
    training_size: Optional[int]
    test_size: Optional[int]
    run_at: datetime
    notes: Optional[str]

    class Config:
        from_attributes = True

class HyperparamUpdate(BaseModel):
    params: dict