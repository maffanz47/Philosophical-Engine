# Philosophical-Engine

Production-oriented ML system for philosophical text analysis.

## Implemented capabilities

- Multi-class classification with PyTorch ANN (Fast and Pro tiers).
- Regression for reading complexity index.
- KNN recommendation with cosine similarity.
- K-Means clustering and PCA 2D projection for report artifacts.
- Prefect workflow for ingestion, preprocessing, training, reporting, and notifications.
- FastAPI `/predict` endpoint supporting JSON text and `.txt` upload.

## Run locally

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

## Run React frontend

In a second terminal:

```powershell
cd .\frontend
npm install
npm run dev
```

Then open `http://localhost:5173` while the API is running on port `8000`.

## Run workflow

```powershell
.\.venv\Scripts\python.exe workflows/main_flow.py
```

## Test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## One-command demo

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1
```
