# ──────────────────────────────────────────────────────────────────────────
# Dockerfile — Philosophical Text Engine API
# Base: python:3.10-slim  (minimal attack surface)
# Exposes port 8000 via uvicorn
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Metadata
LABEL maintainer="maffanz47" \
      description="Philosophical Text Engine — FastAPI inference server"

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ────────────────────────────────────────────────────
# build-essential needed for scikit-learn C extensions on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Create a non-root user for security ───────────────────────────────────
RUN useradd --create-home appuser
WORKDIR /app

# ── Install Python dependencies first (layer caching) ─────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir fastapi uvicorn[standard] pydantic

# ── Copy application source ────────────────────────────────────────────────
COPY api.py          .
COPY engine_core.py  .
COPY taxonomy.py     .
COPY preprocessing.py .
COPY models_supervised.py .
COPY models_unsupervised.py .
COPY ingestion.py    .
COPY validation.py   .

# ── Copy pre-trained artifacts ─────────────────────────────────────────────
# Run `python train.py` locally first, then `docker build`.
COPY engine_artifacts/ ./engine_artifacts/

# Switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# ── Health check ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Runtime ────────────────────────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
