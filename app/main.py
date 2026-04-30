import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import structlog

# Routers
from app.routers import classify, regression, embeddings, recommend, timeseries, clusters, associations

# Preload Models
from ml.classification.school_classifier import load_best_model as load_classifier
from ml.regression.influence_predictor import load_best_model as load_regressor
from ml.recommendation.recommender import load_resources as load_recommender

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading models...")
    load_classifier()
    load_regressor()
    load_recommender()
    app.state.models_loaded = ["school_classifier", "influence_predictor", "recommender"]
    app.state.startup_time = time.time()
    logger.info("Models loaded successfully.")
    yield
    # Shutdown
    logger.info("Shutting down Application...")

app = FastAPI(
    title="The Philosophical Engine API",
    description="From ancient text to machine insight — classifying, recommending, and forecasting human philosophy.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Structlog Middleware
@app.middleware("http")
async def structlog_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        duration_ms=round(process_time, 2),
        status_code=response.status_code
    )
    return response

# Error Handlers
@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    logger.error("internal_error", error=str(exc))
    return JSONResponse(status_code=500, content={"message": "Internal Server Error", "details": str(exc)})

# Routers
app.include_router(classify.router, prefix="/api/v1")
app.include_router(regression.router, prefix="/api/v1")
app.include_router(embeddings.router, prefix="/api/v1")
app.include_router(recommend.router, prefix="/api/v1")
app.include_router(timeseries.router, prefix="/api/v1")
app.include_router(clusters.router, prefix="/api/v1")
app.include_router(associations.router, prefix="/api/v1")

from fastapi.staticfiles import StaticFiles
import os

# Serve the frontend
if os.path.isdir("frontend"):
    app.mount("/dashboard", StaticFiles(directory="frontend", html=True), name="dashboard")

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app.state.startup_time
    return {
        "status": "ok",
        "models_loaded": getattr(app.state, "models_loaded", []),
        "uptime_seconds": round(uptime, 2)
    }

@app.get("/models/info", tags=["System"])
async def models_info():
    """Returns metadata about loaded models."""
    return {
        "classification": "LogisticRegression/XGBoost/DistilBERT",
        "regression": "Ridge/GradientBoosting/LightGBM",
        "embeddings": "all-MiniLM-L6-v2 + PCA + UMAP + t-SNE",
        "recommendation": "Hybrid (Content + Collaborative)",
        "timeseries": "Prophet + IsolationForest",
        "clustering": "K-Means / HDBSCAN",
        "association": "Apriori"
    }

import subprocess
import asyncio
import os

PIPELINE_LOG_FILE = "pipeline.log"
current_pipeline_process = None

@app.post("/api/v1/pipeline/run", tags=["System"])
async def run_pipeline(max_books: int = 10):
    """Triggers the Prefect pipeline asynchronously."""
    global current_pipeline_process
    try:
        # Clear log file if it exists
        open(PIPELINE_LOG_FILE, 'w').close()
        
        # Run the pipeline script as a background process to avoid blocking
        log_file = open(PIPELINE_LOG_FILE, 'a')
        current_pipeline_process = subprocess.Popen(
            ["python", "-m", "pipeline.philosophical_pipeline"], 
            stdout=log_file, 
            stderr=subprocess.STDOUT
        )
        return {"message": "Pipeline triggered successfully in the background.", "pid": current_pipeline_process.pid}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pipeline/status", tags=["System"])
async def pipeline_status():
    """Gets the status and logs of the running pipeline."""
    global current_pipeline_process
    
    is_running = False
    if current_pipeline_process is not None:
        is_running = current_pipeline_process.poll() is None
        
    logs = ""
    if os.path.exists(PIPELINE_LOG_FILE):
        with open(PIPELINE_LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            logs = "".join(lines[-200:])
            
    return {
        "is_running": is_running,
        "logs": logs
    }
