from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers.associations import router as associations_router
from app.routers.classify import router as classify_router
from app.routers.clusters import router as clusters_router
from app.routers.embeddings import router as embeddings_router
from app.routers.recommend import router as recommend_router
from app.routers.regression import router as regression_router
from app.routers.timeseries import router as timeseries_router


def _configure_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )


def _build_app_state() -> dict[str, Any]:
    # Phase 1: no trained models yet. Later phases will load ML artifacts here.
    return {
        "models_loaded": [],
        "model_metadata": {},
        "start_time_monotonic": time.monotonic(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    app.state.philosophical_engine = _build_app_state()
    logger = structlog.get_logger()

    logger.info("lifespan_start", models_loaded=app.state.philosophical_engine["models_loaded"])
    try:
        yield
    finally:
        logger.info("lifespan_end")


app = FastAPI(
    title="Philosophical Engine",
    version="0.1.0",
    lifespan=lifespan,
)

origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
origins = [o.strip() for o in origins_env.split(",")] if origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def structured_request_logging(request: Request, call_next):
    logger = structlog.get_logger()
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        status_code = response.status_code if response is not None else None
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            status_code=status_code,
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "message": "Request validation failed.",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def internal_exception_handler(request: Request, exc: Exception):
    logger = structlog.get_logger()
    logger.error("unhandled_exception", path=request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "message": "An unexpected error occurred.",
            }
        },
    )


@app.get("/health")
def health():
    state = app.state.philosophical_engine
    uptime_seconds = int(time.monotonic() - state["start_time_monotonic"])
    return {
        "status": "ok",
        "models_loaded": state["models_loaded"],
        "uptime_seconds": uptime_seconds,
    }


@app.get("/models/info")
def models_info():
    state = app.state.philosophical_engine
    return {
        "models": state["model_metadata"],
    }


# Routers (versioned)
api_prefix = "/api/v1"
app.include_router(classify_router, prefix=api_prefix)
app.include_router(regression_router, prefix=api_prefix)
app.include_router(embeddings_router, prefix=api_prefix)
app.include_router(recommend_router, prefix=api_prefix)
app.include_router(timeseries_router, prefix=api_prefix)
app.include_router(clusters_router, prefix=api_prefix)
app.include_router(associations_router, prefix=api_prefix)
