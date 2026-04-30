from __future__ import annotations

import os
from datetime import timedelta

from prefect import flow, task
from prefect.logging import get_run_logger


@task(retries=3, retry_delay_seconds=30)
def scrape_gutenberg_flow(max_books: int = 500) -> dict[str, int]:
    logger = get_run_logger()
    logger.info("scrape_gutenberg_flow_start", max_books=max_books)
    # Phase 2+ will call scraper/gutenberg_scraper.py
    return {"max_books": max_books}


@task(retries=3, retry_delay_seconds=30)
def preprocess_flow() -> dict[str, str]:
    logger = get_run_logger()
    logger.info("preprocess_flow_start")
    # Phase 2+ will compute structured features + save embeddings inputs
    return {"status": "ok"}


@task(retries=3, retry_delay_seconds=30)
def train_all_models_flow() -> dict[str, str]:
    logger = get_run_logger()
    logger.info("train_all_models_flow_start")
    # Phase 2+ will train 7 model modules and log to MLflow
    return {"status": "ok"}


@task(retries=3, retry_delay_seconds=30)
def evaluate_flow() -> dict[str, str]:
    logger = get_run_logger()
    logger.info("evaluate_flow_start")
    # Phase 2+ will run DeepChecks and compare against previous MLflow run
    return {"status": "ok"}


@task(retries=3, retry_delay_seconds=30)
def save_artifacts_flow() -> dict[str, str]:
    logger = get_run_logger()
    logger.info("save_artifacts_flow_start")
    # Phase 2+ will version models and save to /models and /reports
    return {"status": "ok"}


def _discord_webhook_url() -> str | None:
    url = os.getenv("DISCORD_WEBHOOK_URL")
    return url or None


@flow(name="philosophical_pipeline")
def philosophical_pipeline(max_books: int = 500) -> dict[str, str]:
    """
    Master Prefect flow (Phase 1 skeleton).

    Phase 2+ will:
      - orchestrate scraping, preprocessing, training, deepchecks evaluation, and artifact saving
      - implement failure/success Discord webhooks via env var DISCORD_WEBHOOK_URL
    """
    _ = scrape_gutenberg_flow(max_books=max_books)
    _ = preprocess_flow()
    _ = train_all_models_flow()
    _ = evaluate_flow()
    _ = save_artifacts_flow()
    return {"status": "ok"}


if __name__ == "__main__":
    philosophical_pipeline()
