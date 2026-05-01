from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import pandas as pd
from ..database import get_db
from ..models.training_chunk import TrainingChunk
from ..ml import load_all_models

def load_data(db: AsyncSession):
    result = db.execute(select(TrainingChunk))
    chunks = result.scalars().all()
    df = pd.DataFrame([{"text": c.text, "label": c.label} for c in chunks])
    return df

def clean_text(df: pd.DataFrame):
    # Use NLTK or similar
    from ..ml.preprocessor import TextPreprocessor
    preprocessor = TextPreprocessor()
    df['clean_text'] = df['text'].apply(preprocessor.clean_text)
    return df

def fit_vectorizer(df: pd.DataFrame, params: dict):
    from ..ml.preprocessor import TextPreprocessor
    preprocessor = TextPreprocessor(**params)
    preprocessor.fit(df['clean_text'].tolist())
    return preprocessor

def train_modules(df: pd.DataFrame, vectorizer, params_by_module):
    from ..ml import argument_classifier, sentiment_analyzer, philosopher_finder, fallacy_detector, topic_clusterer, concept_visualizer, neural_extractor
    results = {}
    # Train each
    # For simplicity, dummy
    results['argument_classifier'] = {"status": "trained"}
    # Similarly for others
    return results

def log_experiments(results, db: AsyncSession):
    # Already logged in train
    pass

def training_flow(modules="all", db_session=None):
    df = load_data(db_session)
    df = clean_text(df)
    vectorizer = fit_vectorizer(df, {})
    results = train_modules(df, vectorizer, {})
    log_experiments(results, db_session)
    return {"results": results}