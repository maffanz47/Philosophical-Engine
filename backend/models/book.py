from sqlalchemy import Column, String, Integer, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
import uuid
from database import Base

class Book(Base):
    __tablename__ = "books"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(500), nullable=False)
    philosopher = Column(String(200), nullable=False)
    status = Column(String(50), default="uploaded")
    page_count = Column(Integer, nullable=True)
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=func.now())
    processed_at = Column(DateTime, nullable=True)