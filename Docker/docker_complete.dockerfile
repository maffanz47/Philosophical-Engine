# ──────────────────────────────────────────────────────────────────────────────
# docker_complete — Philosophical Text Engine
# Purpose : A multi-stage Dockerfile that performs model training during the 
#           build process and serves the FastAPI API in the final image.
#
# Build (from project root):
#   docker build -f Docker/docker_complete -t philo-complete .
#
# Run:
#   docker run --rm -p 8000:8000 philo-complete
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Training ─────────────────────────────────────────────────────────
FROM python:3.10-slim AS trainer

LABEL maintainer="maffanz47" \
      description="Philosophical Text Engine — Training Stage"

# Python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (build-essential for scikit-learn C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy training source files
COPY . .

# Run model training (requires network for Gutenberg scraping)
# This will generate the 'engine_artifacts/' directory
RUN python train.py


# ── Stage 2: Serving ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS final

LABEL maintainer="maffanz47" \
      description="Philosophical Text Engine — Unified API + Frontend"

# Python environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ARTIFACTS_PATH=/app/engine_artifacts

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "fastapi[standard]" uvicorn[standard] pydantic aiofiles

# Copy application source
COPY . .

# Copy the trained artifacts from the trainer stage
COPY --from=trainer /app/engine_artifacts ./engine_artifacts

# Frontend: Serve the single HTML file as a static file at GET /
RUN mkdir -p ./static && cp docs/index.html ./static/index.html

# Non-root user for security
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

# Health check (ping the /health endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Runtime
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
