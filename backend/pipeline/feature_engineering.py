from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..models.training_chunk import TrainingChunk
from ..ml.preprocessor import TextPreprocessor

def fit_tfidf_on_corpus(db: AsyncSession):
    result = db.execute(select(TrainingChunk))
    texts = [c.text for c in result.scalars().all()]
    preprocessor = TextPreprocessor()
    preprocessor.fit(texts)
    return preprocessor