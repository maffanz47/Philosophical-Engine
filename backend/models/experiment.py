from sqlalchemy import Column, String, Float, Integer, DateTime, func, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from database import Base

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    module_name = Column(String(100), nullable=False)
    algorithm = Column(String(100), nullable=False)
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    params_json = Column(JSONB, nullable=False)
    training_size = Column(Integer, nullable=True)
    test_size = Column(Integer, nullable=True)
    run_at = Column(DateTime, default=func.now())
    notes = Column(Text, nullable=True)