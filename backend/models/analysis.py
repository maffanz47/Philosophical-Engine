from sqlalchemy import Column, Text, DateTime, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from database import Base

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    input_text = Column(Text, nullable=False)
    results_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=func.now())