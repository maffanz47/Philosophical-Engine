# ──────────────────────────────────────────────────────────────────────────────
# backend.dockerfile — Philosophical Text Engine
# Purpose : Serve the FastAPI inference API (api.py) on port 8000.
#           The single-file frontend (docs/index.html) is served as a
#           static file via FastAPI's StaticFiles mount at GET /.
#
# Prerequisites:
#   • Run model_training.dockerfile first and ensure engine_artifacts/ is
#     populated on the host before building this image.
#
# Build (from project root):
#   docker build -f Docker/backend.dockerfile -t philo-backend .
#
# Run:
#   docker run --rm \
#     --env-file .env \
#     -p 8000:8000 \
#     philo-backend
#
# Frontend: open http://localhost:8000  in your browser.
# API docs: http://localhost:8000/docs
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

LABEL maintainer="maffanz47" \
      description="Philosophical Text Engine — Backend API + Frontend"

# ── Python environment ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_PATH=/app/engine_artifacts

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "fastapi[standard]" uvicorn[standard] pydantic aiofiles

# ── Backend source files ───────────────────────────────────────────────────
COPY api.py              .
COPY engine_core.py      .
COPY taxonomy.py         .
COPY preprocessing.py    .
COPY models_supervised.py .
COPY models_unsupervised.py .
COPY ingestion.py        .
COPY validation.py       .

# ── Frontend (single HTML file) ────────────────────────────────────────────
# Served as a static file at http://localhost:8000/
COPY docs/index.html     ./static/index.html

# ── Pre-trained model artifacts ────────────────────────────────────────────
# Build after running model_training.dockerfile so this directory is populated.
COPY engine_artifacts/   ./engine_artifacts/

# ── Non-root user ──────────────────────────────────────────────────────────
RUN useradd --create-home appuser \
 && chown -R appuser:appuser /app
USER appuser

# ── Health check ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Runtime ────────────────────────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
