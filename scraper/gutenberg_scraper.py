from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
import spacy
from bs4 import BeautifulSoup
from bs4.element import Tag
from textblob import TextBlob

import nltk
from nltk.corpus import stopwords as nltk_stopwords

import pandas as pd


RANDOM_SEED_ENV = "RANDOM_SEED"
DEFAULT_MIN_BOOKS = 500
REQUEST_SLEEP_SECONDS = 2.0
RETRY_COUNT = 3
RETRY_BACKOFF_BASE_SECONDS = 2.0


@dataclass(frozen=True)
class BookRaw:
    gutenberg_id: int
    title: str
    author: str | None
    year: int | None
    subjects: list[str]
    download_count: int | None
    full_text: str


# Mapping based on Gutenberg subject tags (heuristic keyword matching).
SUBJECT_TO_SCHOOL_KEYWORDS: dict[str, list[str]] = {
    "Empiricism": ["empiricism", "skepticism", "experience", "induction", "empirical"],
    "Rationalism": ["rationalism", "reason", "metaphysics", "logic", "deduction"],
    "Existentialism": ["existentialism", "existence", "freedom", "authenticity"],
    "Stoicism": ["stoicism", "stoic"],
    "Idealism": ["idealism", "transcendental", "absolute idealism"],
    "Pragmatism": ["pragmatism", "instrumental", "verification", "truth"],
    "Other": [],
}


def _ensure_nltk_stopwords() -> set[str]:
    try:
        nltk_stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    return set(nltk_stopwords.words("english"))


def _rate_limited_get(session: requests.Session, url: str, *, timeout_seconds: int = 30) -> requests.Response:
    # Simple global rate limiting per process.
    time.sleep(REQUEST_SLEEP_SECONDS)
    return session.get(url, timeout=timeout_seconds)


def _request_with_retries(
    session: requests.Session, url: str, *, timeout_seconds: int = 30
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(RETRY_COUNT):
        try:
            return _rate_limited_get(session, url, timeout_seconds=timeout_seconds)
        except requests.RequestException as exc:
            last_exc = exc
            backoff = RETRY_BACKOFF_BASE_SECONDS**attempt
            time.sleep(backoff)
    # Final attempt failed
    raise RuntimeError(f"Failed to GET {url} after {RETRY_COUNT} retries: {last_exc}")


def _clean_text(text: str) -> str:
    # Normalize whitespace; keep newlines for sentence splitting.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _parse_int_from_text(text: str) -> int | None:
    m = re.search(r"(\d[\d,]*)", text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def _parse_year_from_release_date(release_date_text: str) -> int | None:
    # Example: "Release Date: March 12, 2019"
    m = re.search(r"Release Date:\s*[A-Za-z]+\s+\d{1,2},\s+(\d{4})", release_date_text)
    if m:
        return int(m.group(1))
    # Fallback: any 4-digit year
    m2 = re.search(r"(\d{4})", release_date_text)
    return int(m2.group(1)) if m2 else None


def _extract_book_page_fields(book_soup: BeautifulSoup) -> tuple[str, str | None, int | None, list[str], int | None]:
    # Title
    title_tag = book_soup.select_one("h1.title")
    title = title_tag.get_text(strip=True) if isinstance(title_tag, Tag) else ""

    # Author
    author_tag = book_soup.select_one("a[href^='/ebooks/']")  # fallback: may not always be correct
    author: str | None = None
    # More reliable pattern: the "By" section is often in a div with class "book-title"
    author_meta = book_soup.select_one("a[rel='author']")
    if isinstance(author_meta, Tag):
        author = author_meta.get_text(strip=True)

    # Subjects and Release date fields are on Gutenberg page in "table" blocks.
    subjects: list[str] = []
    download_count: int | None = None
    year: int | None = None

    # Release date / download count appear in various sections; we heuristically scan text.
    page_text = book_soup.get_text("\n", strip=True)

    # Parse release date/year
    year = _parse_year_from_release_date(page_text)

    # Subjects: look for "Subjects" label and parse nearby links.
    # Gutenberg often renders subjects as: <div id="book_subjects"> ... <a>...</a> ...
    subjects_container = book_soup.select_one("#book_subjects")
    if isinstance(subjects_container, Tag):
        for a in subjects_container.select("a"):
            val = a.get_text(strip=True)
            if val:
                subjects.append(val)

    # Download count: "Downloads" sometimes appears near the "downloads" section.
    # We look for a number preceded by "Downloads".
    download_match = re.search(r"Downloads\s*:\s*([\d,]+)", page_text, flags=re.IGNORECASE)
    if download_match:
        download_count = int(download_match.group(1).replace(",", ""))

    return title, author, year, subjects, download_count


def _download_plain_text(session: requests.Session, gutenberg_id: int) -> str:
    # Try UTF8-cached .txt first (common), then fallback.
    # Many Gutenberg ids have cache/epub/ID/pgID.txt.utf8 or pgID.txt.
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt.utf8",
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
    ]
    last_exc: Exception | None = None
    for url in candidates:
        try:
            resp = _request_with_retries(session, url, timeout_seconds=60)
            resp.raise_for_status()
            # Handle encoding errors gracefully
            return resp.content.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to download plain text for gutenberg_id={gutenberg_id}: {last_exc}")


def _school_label_from_subjects(subjects: Iterable[str]) -> str:
    subject_lc = {s.strip().lower() for s in subjects if s and s.strip()}
    for school, keywords in SUBJECT_TO_SCHOOL_KEYWORDS.items():
        if not keywords:
            continue
        if any(any(kw in s for kw in keywords) for s in subject_lc):
            return school
    return "Other"


def _derive_era_label(year: int | None) -> str:
    if year is None:
        return "Contemporary"
    if year <= 500:
        return "Ancient"
    if year <= 1400:
        return "Medieval"
    if year <= 1600:
        return "Renaissance"
    if year <= 1800:
        return "Enlightenment"
    if year <= 1950:
        return "Modern"
    return "Contemporary"


def _round_to_nearest_decade(year: int | None) -> int | None:
    if year is None:
        return None
    return int(round(year / 10.0) * 10)


def _top_noun_concepts(nlp: Any, text: str, *, top_k: int = 10, extra_stopwords: set[str] | None = None) -> list[str]:
    stop_set = _ensure_nltk_stopwords()
    if extra_stopwords:
        stop_set |= extra_stopwords

    doc = nlp(text)
    freq: dict[str, int] = {}
    for token in doc:
        if token.is_stop:
            continue
        if token.is_alpha is False:
            continue
        if token.lemma_:
            lemma = token.lemma_.lower().strip()
        else:
            lemma = token.text.lower().strip()
        if not lemma or lemma in stop_set:
            continue
        if token.pos_ in {"NOUN", "PROPN"}:
            freq[lemma] = freq.get(lemma, 0) + 1

    # Sort by frequency desc
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [w for w, _ in ranked]


def _extract_features(
    nlp: Any,
    text: str,
    *,
    year: int | None,
) -> dict[str, Any]:
    text = _clean_text(text)
    if not text:
        return {
            "avg_sentence_length": 0.0,
            "vocab_richness": 0.0,
            "sentiment_polarity": 0.0,
            "named_entity_count": 0,
            "top_concepts": [],
            "decade": _round_to_nearest_decade(year),
            "era_label": _derive_era_label(year),
        }

    doc = nlp(text)

    # Sentence length based on tokens per sentence
    sentences = list(doc.sents)
    if not sentences:
        sentences = [doc]

    sentence_lengths = [len(list(s)) for s in sentences if len(list(s)) > 0]
    avg_sentence_length = float(sum(sentence_lengths) / max(1, len(sentence_lengths)))

    tokens = [t for t in doc if not t.is_space and t.is_alpha]
    total_tokens = max(1, len(tokens))
    unique_tokens = len({t.lemma_.lower() for t in tokens if t.lemma_})
    vocab_richness = float(unique_tokens / total_tokens)

    sentiment_polarity = float(TextBlob(text).sentiment.polarity)
    named_entity_count = int(len(doc.ents))

    top_concepts = _top_noun_concepts(nlp, text)

    decade = _round_to_nearest_decade(year)
    era_label = _derive_era_label(year)

    return {
        "avg_sentence_length": avg_sentence_length,
        "vocab_richness": vocab_richness,
        "sentiment_polarity": sentiment_polarity,
        "named_entity_count": named_entity_count,
        "top_concepts": top_concepts,
        "decade": decade,
        "era_label": era_label,
    }


def _parse_search_results(search_soup: BeautifulSoup) -> list[int]:
    # Search results contain links like /ebooks/{id}
    ids: set[int] = set()
    for a in search_soup.select("a[href^='/ebooks/']"):
        href = a.get("href") or ""
        m = re.match(r"^/ebooks/(\d+)/?$", href)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def scrape_gutenberg_philosophy(
    *,
    query: str = "philosophy",
    min_books: int = DEFAULT_MIN_BOOKS,
    out_raw_dir: str | Path = "data/raw",
    out_processed_csv: str | Path = "data/processed/philosophy_corpus.csv",
    cache: bool = True,
    limit_books: int | None = None,
) -> None:
    """
    Scrape Project Gutenberg for the query "philosophy", paginate through search results,
    and build a processed CSV with derived NLP features.

    Saves raw per-book JSON to data/raw/{gutenberg_id}.json and a CSV to
    data/processed/philosophy_corpus.csv.

    Implements:
      - rate limiting (1 request per 2 seconds)
      - retry logic (3 retries with exponential backoff)
      - local cache so already-downloaded texts are skipped
    """
    out_raw_dir = Path(out_raw_dir)
    out_processed_csv = Path(out_processed_csv)
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    out_processed_csv.parent.mkdir(parents=True, exist_ok=True)

    # NLP
    nlp = spacy.load("en_core_web_sm", disable=["textcat"])
    nlp.max_length = 2_000_000

    session = requests.Session()
    base_search_url = "https://www.gutenberg.org/ebooks/search/"
    start_index = 0
    collected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    # Iterate until we hit min_books (or optional limit_books)
    while True:
        if limit_books is not None and len(seen_ids) >= limit_books:
            break
        if len(seen_ids) >= min_books:
            break

        params = f"?query={requests.utils.quote(query)}&submit_search=Search&start_index={start_index}"
        search_url = base_search_url + params

        resp = _request_with_retries(session, search_url, timeout_seconds=30)
        resp.raise_for_status()
        search_soup = BeautifulSoup(resp.text, "lxml")
        book_ids = _parse_search_results(search_soup)

        if not book_ids:
            # No results; stop to avoid infinite loop
            break

        for gutenberg_id in book_ids:
            if limit_books is not None and len(seen_ids) >= limit_books:
                break
            if gutenberg_id in seen_ids:
                continue

            seen_ids.add(gutenberg_id)
            raw_path = out_raw_dir / f"{gutenberg_id}.json"

            if cache and raw_path.exists():
                try:
                    data = json.loads(raw_path.read_text(encoding="utf-8"))
                    collected.append(data)
                    continue
                except Exception:  # noqa: BLE001
                    # If cache is corrupted, re-download.
                    pass

            # Fetch book metadata page
            book_url = f"https://www.gutenberg.org/ebooks/{gutenberg_id}"
            book_resp = _request_with_retries(session, book_url, timeout_seconds=30)
            book_resp.raise_for_status()
            book_soup = BeautifulSoup(book_resp.text, "lxml")

            title, author, year, subjects, download_count = _extract_book_page_fields(book_soup)

            full_text = _download_plain_text(session, gutenberg_id)
            full_text = full_text.strip()

            book_raw = BookRaw(
                gutenberg_id=gutenberg_id,
                title=title,
                author=author,
                year=year,
                subjects=subjects,
                download_count=download_count,
                full_text=full_text,
            )

            # Derive features from full_text
            derived = _extract_features(nlp, book_raw.full_text, year=book_raw.year)

            record: dict[str, Any] = {
                "gutenberg_id": book_raw.gutenberg_id,
                "title": book_raw.title,
                "author": book_raw.author,
                "year": book_raw.year,
                "subjects": book_raw.subjects,
                "download_count": book_raw.download_count,
                "avg_sentence_length": derived["avg_sentence_length"],
                "vocab_richness": derived["vocab_richness"],
                "sentiment_polarity": derived["sentiment_polarity"],
                "named_entity_count": derived["named_entity_count"],
                "top_concepts": derived["top_concepts"],
                "decade": derived["decade"],
                "era_label": derived["era_label"],
                "school_label": _school_label_from_subjects(book_raw.subjects),
                "full_text": book_raw.full_text,
            }

            raw_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            collected.append(record)

        # Gutenberg pages often show 25 results; we increment by 25.
        start_index += 25

    # Persist processed CSV (without full_text)
    rows_for_csv: list[dict[str, Any]] = []
    for r in collected:
        row = dict(r)
        row.pop("full_text", None)
        rows_for_csv.append(row)

    df = pd.DataFrame(rows_for_csv)
    df.to_csv(out_processed_csv, index=False, encoding="utf-8")

    print(f"Scraped {len(rows_for_csv)} books into {out_processed_csv}")


if __name__ == "__main__":
    # Basic local run
    scrape_gutenberg_philosophy()
