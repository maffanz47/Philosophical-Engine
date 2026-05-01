"""FastAPI service exposing prediction endpoint."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from src.service import inference_service

app = FastAPI(title="Philosophical Text Engine API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw philosophical input text")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(http_request: Request, file: UploadFile | None = File(default=None)) -> dict:
    """
    Predict school, confidence, complexity, and recommendations.

    Supports either JSON body with `text` or `.txt` upload.
    """
    text: str | None = None

    content_type = (http_request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        body = await http_request.json()
        parsed = PredictRequest.model_validate(body)
        text = parsed.text.strip()
    elif file is not None:
        if not file.filename or not file.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt uploads are supported.")
        content = await file.read()
        text = content.decode("utf-8", errors="ignore").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if not text:
        raise HTTPException(status_code=400, detail="Provide either JSON text or a .txt file.")

    result = inference_service.predict(text)
    return result.to_dict()
