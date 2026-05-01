"""
data_loader.py
==============
Automated ingestion of philosophical texts from the Project Gutenberg API
(via RapidAPI). Implements retry logic, school-to-book mapping, and local
caching so the expensive network calls are only made once.

Book → Philosophical School mapping
-------------------------------------
| School           | Author          | Gutenberg ID |
|------------------|-----------------|--------------|
| Stoicism         | Marcus Aurelius | 2680         |
| Stoicism         | Epictetus       | 4135         |
| Existentialism   | Nietzsche       | 1998         |
| Existentialism   | Kierkegaard     | 4057         |
| Islamic Phil.    | Al-Ghazali      | 35497        |
| Rationalism      | Descartes       | 59           |
| Rationalism      | Spinoza         | 3800         |
| Empiricism       | John Locke      | 10615        |
| Empiricism       | David Hume      | 4705         |
| Idealism         | Plato           | 1497         |
| Idealism         | Kant            | 4280         |
"""

import os
import time
import json
import logging
import requests
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("data_loader")

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
RAPIDAPI_HOST = "gutendex.p.rapidapi.com"
RAPIDAPI_KEY = os.getenv("project_gutenber_api_from_rapidapi.com", "")

HEADERS = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST,
}

# Raw-text mirror that serves plain UTF-8 — no API key required.
# Format: https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt
GUTENBERG_TEXT_URL = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Book → School mapping
# ---------------------------------------------------------------------------
BOOK_SCHOOL_MAP: dict[int, dict] = {
    2680:  {"school": "Stoicism",        "author": "Marcus Aurelius", "title": "Meditations"},
    4135:  {"school": "Stoicism",        "author": "Epictetus",       "title": "Discourses"},
    1998:  {"school": "Existentialism",  "author": "Nietzsche",       "title": "Beyond Good and Evil"},
    4057:  {"school": "Existentialism",  "author": "Kierkegaard",     "title": "Either Or"},
    35497: {"school": "Islamic Philosophy", "author": "Al-Ghazali",  "title": "The Alchemy of Happiness"},
    59:    {"school": "Rationalism",     "author": "Descartes",       "title": "Meditations on First Philosophy"},
    3800:  {"school": "Rationalism",     "author": "Spinoza",         "title": "Ethics"},
    10615: {"school": "Empiricism",      "author": "John Locke",      "title": "An Essay Concerning Human Understanding"},
    4705:  {"school": "Empiricism",      "author": "David Hume",      "title": "An Enquiry Concerning Human Understanding"},
    1497:  {"school": "Idealism",        "author": "Plato",           "title": "The Republic"},
    4280:  {"school": "Idealism",        "author": "Kant",            "title": "Critique of Pure Reason"},
}

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
def fetch_with_retry(url: str, headers: Optional[dict] = None,
                     max_retries: int = 3, backoff: float = 2.0) -> Optional[requests.Response]:
    """
    Perform a GET request with exponential-backoff retry logic.

    Args:
        url:         Target URL.
        headers:     Optional HTTP headers.
        max_retries: Maximum number of attempts.
        backoff:     Base seconds between retries (doubles each attempt).

    Returns:
        requests.Response on success, None on total failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            wait = backoff ** attempt
            logger.warning(
                "Attempt %d/%d failed for %s — %s. Retrying in %.1fs…",
                attempt, max_retries, url, exc, wait,
            )
            if attempt < max_retries:
                time.sleep(wait)
    logger.error("All %d attempts exhausted for %s", max_retries, url)
    return None


# ---------------------------------------------------------------------------
# Book metadata lookup (Gutendex via RapidAPI)
# ---------------------------------------------------------------------------
def get_book_metadata(book_id: int) -> Optional[dict]:
    """
    Fetch metadata for a single book from Gutendex API on RapidAPI.

    Args:
        book_id: Project Gutenberg numeric ID.

    Returns:
        Parsed JSON dict or None.
    """
    url = f"https://{RAPIDAPI_HOST}/books/{book_id}"
    response = fetch_with_retry(url, headers=HEADERS)
    if response is None:
        return None
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        logger.error("JSON decode error for book %d: %s", book_id, exc)
        return None


# ---------------------------------------------------------------------------
# Raw text downloader
# ---------------------------------------------------------------------------
def download_book_text(book_id: int, force: bool = False) -> Optional[Path]:
    """
    Download the plain-text content of a Gutenberg book and cache it locally.

    Args:
        book_id: Project Gutenberg numeric ID.
        force:   Re-download even if the file already exists.

    Returns:
        Path to the saved .txt file, or None on failure.
    """
    info = BOOK_SCHOOL_MAP.get(book_id, {})
    safe_title = info.get("title", str(book_id)).replace(" ", "_").replace("/", "-")
    dest = DATA_DIR / f"{book_id}_{safe_title}.txt"

    if dest.exists() and not force:
        logger.info("Cache hit: %s (skipping download)", dest.name)
        return dest

    url = GUTENBERG_TEXT_URL.format(book_id=book_id)
    logger.info("Downloading book %d from %s …", book_id, url)
    response = fetch_with_retry(url)
    if response is None:
        return None

    dest.write_bytes(response.content)
    logger.info("Saved %d bytes → %s", len(response.content), dest)
    return dest


# ---------------------------------------------------------------------------
# Public API: ingest all books
# ---------------------------------------------------------------------------
def ingest_all_books(force: bool = False) -> list[dict]:
    """
    Download every book in BOOK_SCHOOL_MAP and return a manifest list.

    Each manifest entry contains:
        {
          "book_id": int,
          "school":  str,
          "author":  str,
          "title":   str,
          "path":    str   (absolute path to .txt file)
        }

    Args:
        force: Re-download even if local copy exists.

    Returns:
        List of manifest dicts for successfully downloaded books.
    """
    manifest = []
    for book_id, info in BOOK_SCHOOL_MAP.items():
        path = download_book_text(book_id, force=force)
        if path is not None:
            manifest.append({
                "book_id": book_id,
                "school":  info["school"],
                "author":  info["author"],
                "title":   info["title"],
                "path":    str(path),
            })
        else:
            logger.warning("Skipping book %d (%s) — download failed.", book_id, info.get("title"))

    logger.info("Ingestion complete. %d/%d books downloaded.", len(manifest), len(BOOK_SCHOOL_MAP))

    # Persist manifest as JSON for downstream tasks
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest saved → %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# Entry point (standalone test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    result = ingest_all_books()
    print(f"\n✓ Downloaded {len(result)} books.")
