from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from ..database import get_db
from ..models.book import Book
from ..models.training_chunk import TrainingChunk
from ..models.experiment import Experiment
from ..models.user import User
from ..core.permissions import require_admin
from ..pipeline.pdf_processor import extract_chunks_from_pdf
from ..ml import (argument_classifier, sentiment_analyzer, complexity_scorer,
                  philosopher_finder, fallacy_detector, topic_clusterer,
                  concept_visualizer, neural_extractor, socratic_agent)
from ..schemas.admin import BookOut, ExperimentOut
from pydantic import BaseModel
from typing import List
import uuid
from pathlib import Path

router = APIRouter()

@router.post("/books/upload", response_model=BookOut)
async def upload_book(
    file: UploadFile = File(...),
    philosopher: str = Form(...),
    category: str = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(require_admin)
):
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    filepath = uploads_dir / f"{uuid.uuid4()}_{file.filename}"
    with open(filepath, "wb") as f:
        f.write(await file.read())
    
    book = Book(filename=file.filename, philosopher=philosopher)
    db.add(book)
    await db.commit()
    await db.refresh(book)
    return book

@router.get("/books", response_model=List[BookOut])
async def get_books(db: AsyncSession = Depends(get_db), current_user = Depends(require_admin)):
    result = await db.execute(select(Book))
    books = result.scalars().all()
    return books

@router.post("/books/{book_id}/process")
async def process_book(book_id: str, db: AsyncSession = Depends(get_db), current_user = Depends(require_admin)):
    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    filepath = Path("uploads") / book.filename  # adjust
    chunk_count = await extract_chunks_from_pdf(book_id, str(filepath), book.philosopher, db)
    book.status = "processed"
    book.chunk_count = chunk_count
    await db.commit()
    return {"chunk_count": chunk_count}

@router.delete("/books/{book_id}")
async def delete_book(book_id: str, db: AsyncSession = Depends(get_db), current_user = Depends(require_admin)):
    result = await db.execute(select(Book).where(Book.id == book_id))
    book = result.scalar_one_or_none()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    # Delete chunks
    await db.execute(select(TrainingChunk).where(TrainingChunk.book_id == book_id).delete())
    await db.delete(book)
    await db.commit()
    return {"deleted": True}

@router.post("/train/{module_name}")
async def train_module(module_name: str, params: dict = {}, db: AsyncSession = Depends(get_db), current_user = Depends(require_admin)):
    if module_name == "argument_classifier":
        result = await argument_classifier.train(db, params)
    elif module_name == "sentiment_analyzer":
        result = await sentiment_analyzer.train(db, params)
    elif module_name == "philosopher_finder":
        result = await philosopher_finder.train(db, params)
    elif module_name == "fallacy_detector":
        result = await fallacy_detector.train(db, params)
    elif module_name == "topic_clusterer":
        result = await topic_clusterer.train(db, params)
    elif module_name == "concept_visualizer":
        result = await concept_visualizer.train(db, params)
    elif module_name == "neural_extractor":
        result = await neural_extractor.train(db, params)
    else:
        raise HTTPException(status_code=400, detail="Invalid module")
    return result

@router.post("/train/all")
async def train_all(db: AsyncSession = Depends(get_db), current_user = Depends(require_admin)):
    # Train all
    results = {}
    results["argument_classifier"] = await argument_classifier.train(db, {})
    results["sentiment_analyzer"] = await sentiment_analyzer.train(db, {})
    results["philosopher_finder"] = await philosopher_finder.train(db, {})
    results["fallacy_detector"] = await fallacy_detector.train(db, {})
    results["topic_clusterer"] = await topic_clusterer.train(db, {})
    results["concept_visualizer"] = await concept_visualizer.train(db, {})
    results["neural_extractor"] = await neural_extractor.train(db, {})
    return results

@router.get("/hyperparams")
async def get_hyperparams(current_user = Depends(require_admin)):
    # Return default params
    return {
        "argument_classifier": {"lr": {"C": 1.0}, "svm": {"C": 1.0}},
        "sentiment_analyzer": {"lr": {"C": 1.0}, "knn": {"n_neighbors": 5}},
        "philosopher_finder": {"svm": {"C": 1.0}, "knn": {"n_neighbors": 5}},
        "fallacy_detector": {"nb": {"alpha": 1.0}},
        "topic_clusterer": {"kmeans": {"n_clusters": 5}},
        "concept_visualizer": {},
        "neural_extractor": {"learning_rate": 0.001, "epochs": 30},
        "socratic_agent": {"alpha": 0.1, "gamma": 0.9}
    }

@router.put("/hyperparams/{module}")
async def update_hyperparams(module: str, params: dict, current_user = Depends(require_admin)):
    # Save to DB or file, for now dummy
    return {"updated": True}

@router.get("/models")
async def get_models(current_user = Depends(require_admin)):
    # List saved models
    models_dir = Path("saved_models")
    models = [f.name for f in models_dir.iterdir() if f.is_file()]
    return {"models": models}

@router.delete("/models/{module_name}")
async def delete_model(module_name: str, current_user = Depends(require_admin)):
    # Delete files
    models_dir = Path("saved_models")
    for f in models_dir.glob(f"{module_name}*"):
        f.unlink()
    return {"deleted": True}

@router.get("/experiments", response_model=List[ExperimentOut])
async def get_experiments(module: str = None, limit: int = 50, db: AsyncSession = Depends(get_db), current_user = Depends(require_admin)):
    query = select(Experiment)
    if module:
        query = query.where(Experiment.module_name == module)
    result = await db.execute(query.order_by(Experiment.run_at.desc()).limit(limit))
    experiments = result.scalars().all()
    return experiments

@router.get("/users")
async def get_users(db: AsyncSession = Depends(get_db), current_user = Depends(require_admin)):
    result = await db.execute(select(User))
    users = result.scalars().all()
    return [{"id": str(u.id), "email": u.email, "role": u.role, "created_at": u.created_at.isoformat()} for u in users]

@router.get("/users/{user_id}/history")
async def get_user_history(user_id: str, db: AsyncSession = Depends(get_db), current_user = Depends(require_admin)):
    result = await db.execute(select(Analysis).where(Analysis.user_id == user_id))
    analyses = result.scalars().all()
    return [{"id": str(a.id), "input_text": a.input_text[:100], "created_at": a.created_at.isoformat()} for a in analyses]

@router.post("/pipeline/run")
async def run_pipeline(db: AsyncSession = Depends(get_db), current_user = Depends(require_admin)):
    from ..pipeline.prefect_pipeline import training_flow
    # Run flow
    result = training_flow(db_session=db)
    return {"status": "running"}

@router.get("/pipeline/status")
async def pipeline_status(current_user = Depends(require_admin)):
    return {"status": "completed"}  # dummy

@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db), current_user = Depends(require_admin)):
    total_chunks = await db.execute(select(func.count(TrainingChunk.id)))
    total_books = await db.execute(select(func.count(Book.id)))
    chunks_per_philosopher = {}  # dummy
    return {
        "total_chunks": total_chunks.scalar(),
        "total_books": total_books.scalar(),
        "chunks_per_philosopher": chunks_per_philosopher
    }