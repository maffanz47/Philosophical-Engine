from sqlalchemy import Column, String, Text, Integer, DateTime, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid
from database import Base

class TrainingChunk(Base):
    __tablename__ = "training_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    book_id = Column(UUID(as_uuid=True), ForeignKey("books.id"), nullable=False)
    text = Column(Text, nullable=False)
    label = Column(String(200), nullable=False)
    category = Column(String(100), nullable=True)
    chunk_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now())