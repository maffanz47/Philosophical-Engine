from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models.analysis import Analysis
from ..models.user import User
from ..core.permissions import require_user
from ..ml import (argument_classifier, sentiment_analyzer, complexity_scorer,
                  philosopher_finder, fallacy_detector, topic_clusterer,
                  concept_visualizer, neural_extractor, socratic_agent)
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

class AnalyzeRequest(BaseModel):
    text: str
    modules: Optional[List[str]] = None

class AnalyzeResponse(BaseModel):
    results: dict

class HistoryResponse(BaseModel):
    id: str
    input_text: str
    created_at: str

class SocraticRequest(BaseModel):
    message: str
    session_state: dict

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest, db: AsyncSession = Depends(get_db), current_user: User = Depends(require_user)):
    modules = request.modules or ["all"]
    results = {}

    if "argument_classifier" in modules or "all" in modules:
        results["argument_classifier"] = argument_classifier.predict(request.text)
    if "sentiment_analyzer" in modules or "all" in modules:
        results["sentiment_analyzer"] = sentiment_analyzer.predict(request.text)
    if "complexity_scorer" in modules or "all" in modules:
        results["complexity_scorer"] = complexity_scorer.predict(request.text)
    if "philosopher_finder" in modules or "all" in modules:
        results["philosopher_finder"] = philosopher_finder.predict(request.text)
    if "fallacy_detector" in modules or "all" in modules:
        results["fallacy_detector"] = fallacy_detector.predict(request.text)
    if "topic_clusterer" in modules or "all" in modules:
        results["topic_clusterer"] = topic_clusterer.predict(request.text)
    if "concept_visualizer" in modules or "all" in modules:
        results["concept_visualizer"] = concept_visualizer.predict(request.text)
    if "neural_extractor" in modules or "all" in modules:
        results["neural_extractor"] = neural_extractor.predict(request.text)

    # Save to DB
    analysis = Analysis(user_id=current_user.id, input_text=request.text, results_json=results)
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    return AnalyzeResponse(results=results)

@router.get("/history")
async def get_history(page: int = 1, limit: int = 10, db: AsyncSession = Depends(get_db), current_user: User = Depends(require_user)):
    offset = (page - 1) * limit
    result = await db.execute(
        select(Analysis).where(Analysis.user_id == current_user.id).order_by(Analysis.created_at.desc()).offset(offset).limit(limit)
    )
    analyses = result.scalars().all()
    return [HistoryResponse(id=str(a.id), input_text=a.input_text[:100] + "...", created_at=a.created_at.isoformat()) for a in analyses]

@router.get("/history/{analysis_id}")
async def get_analysis(analysis_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(require_user)):
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == current_user.id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"id": str(analysis.id), "input_text": analysis.input_text, "results": analysis.results_json, "created_at": analysis.created_at.isoformat()}

@router.post("/socratic")
async def socratic_question(request: SocraticRequest, current_user: User = Depends(require_user)):
    result = socratic_agent.predict(request.session_state)
    return result