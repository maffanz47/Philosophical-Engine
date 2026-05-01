"""Data ingestion utilities for philosophical text corpora."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

GUTENBERG_MIRROR_URL = "https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"


@dataclass
class IngestionConfig:
    """Runtime options for resilient text ingestion."""

    retries: int = 3
    retry_delay_seconds: float = 1.5
    timeout_seconds: float = 20.0
    chunk_size_words: int = 400
    min_chunk_words: int = 300
    max_chunk_words: int = 500


def build_gutenberg_url(book_id: int) -> str:
    """Return a canonical Gutenberg plain-text URL."""
    return GUTENBERG_MIRROR_URL.format(book_id=book_id)


def fetch_text_with_retries(url: str, config: Optional[IngestionConfig] = None) -> str:
    """Fetch a remote text document with retry logic."""
    cfg = config or IngestionConfig()
    last_error: Optional[Exception] = None

    for attempt in range(1, cfg.retries + 1):
        try:
            response = requests.get(url, timeout=cfg.timeout_seconds)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < cfg.retries:
                time.sleep(cfg.retry_delay_seconds)

    raise RuntimeError(f"Failed to fetch text after {cfg.retries} attempts: {url}") from last_error


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Remove standard Gutenberg headers and footers from downloaded text."""
    start_pattern = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*", re.IGNORECASE)
    end_pattern = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*", re.IGNORECASE)

    start_match = start_pattern.search(raw_text)
    end_match = end_pattern.search(raw_text)

    start_index = start_match.end() if start_match else 0
    end_index = end_match.start() if end_match else len(raw_text)
    return raw_text[start_index:end_index].strip()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def segment_text(text: str, min_words: int = 300, max_words: int = 500, target_words: int = 400) -> List[str]:
    """
    Segment text into approximately 300-500 word chunks.

    The function prefers target-sized windows and attaches short tails
    to the previous chunk to avoid creating tiny fragments.
    """
    if min_words > max_words:
        raise ValueError("min_words must be <= max_words")
    if not min_words <= target_words <= max_words:
        raise ValueError("target_words must be between min_words and max_words")

    words = _normalize_whitespace(text).split(" ")
    words = [word for word in words if word]
    if not words:
        return []

    chunks: List[str] = []
    idx = 0
    total = len(words)

    while idx < total:
        remaining = total - idx
        if remaining <= max_words:
            if remaining < min_words and chunks:
                chunks[-1] = f"{chunks[-1]} {' '.join(words[idx:])}".strip()
            else:
                chunks.append(" ".join(words[idx:]))
            break

        end = idx + target_words
        candidate_remaining = total - end
        if 0 < candidate_remaining < min_words:
            end = total - min_words
        if end - idx < min_words:
            end = idx + min_words
        if end - idx > max_words:
            end = idx + max_words

        chunks.append(" ".join(words[idx:end]))
        idx = end

    return chunks


def ingest_gutenberg_book(book_id: int, config: Optional[IngestionConfig] = None) -> List[str]:
    """Download, clean, and segment a Gutenberg book into model-ready chunks."""
    cfg = config or IngestionConfig()
    raw_text = fetch_text_with_retries(build_gutenberg_url(book_id), cfg)
    cleaned_text = strip_gutenberg_boilerplate(raw_text)
    return segment_text(
        cleaned_text,
        min_words=cfg.min_chunk_words,
        max_words=cfg.max_chunk_words,
        target_words=cfg.chunk_size_words,
    )
