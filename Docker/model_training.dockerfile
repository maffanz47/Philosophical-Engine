# ──────────────────────────────────────────────────────────────────────────────
# model_training.dockerfile — Philosophical Text Engine
# Purpose : Run train.py to ingest Gutenberg texts, train all models
#           (SVM Slow/Fast, MLP, K-Means Slow/Fast, PCA) and persist
#           the resulting artifacts to ./engine_artifacts via a volume mount.
#
# Build (from project root):
#   docker build -f Docker/model_training.dockerfile -t philo-train .
#
# Run (mount host ./engine_artifacts so artifacts survive container exit):
#   On Windows (PowerShell):
#     docker run --rm --env-file .env -v "${PWD}/engine_artifacts:/app/engine_artifacts" philo-train
#   On Mac/Linux:
#     docker run --rm --env-file .env -v "$(pwd)/engine_artifacts:/app/engine_artifacts" philo-train
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

LABEL maintainer="maffanz47" \
    description="Philosophical Text Engine — Model Training Pipeline"

# ── Python environment ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ────────────────────────────────────────────────────
# build-essential: needed to compile scikit-learn / scipy C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Training source files ──────────────────────────────────────────────────
# Only the files actually imported by train.py are needed here.
COPY train.py            .
COPY engine_core.py      .
COPY ingestion.py        .
COPY preprocessing.py    .
COPY validation.py       .
COPY models_supervised.py .
COPY models_unsupervised.py .
COPY taxonomy.py         .
COPY notify.py           .

# ── Artifacts output directory ─────────────────────────────────────────────
# Mount your host ./engine_artifacts here at `docker run` time so that
# the trained models persist after the container exits.
RUN mkdir -p /app/engine_artifacts
VOLUME ["/app/engine_artifacts"]

# ── Entry point ───────────────────────────────────────────────────────────
# Train all models and exit.  Notifications (Discord/email) fire automatically
# at the end of train.py if the matching env-vars are present in .env.
CMD ["python", "train.py"]