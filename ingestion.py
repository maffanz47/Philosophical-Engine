"""
ingestion.py — Fault-Tolerant Multi-Source Ingestion Pipeline (v5).

Key changes vs previous:
  • Fixed-Window Scraper: Split text into exactly 100-word chunks, discarding paragraph boundaries.
  • Target quota: 2,500 chunks per Tier-2 category (17,500 total).
  • 70-book library (10 IDs per category) cycled sequentially.
  • Rate limiting: time.sleep(1.5) between every HTTP request.
  • Graceful degradation: if any category falls short of its quota after
    exhausting all 10 sources, the pipeline detects the minimum count
    across all 7 categories and down-samples every category to that
    "lowest common denominator", guaranteeing perfect parity before
    returning data to the vectorizer and DataLoaders.
  • tqdm progress bars at the category and book level.
"""

import re
import time
import requests
from tqdm import tqdm

from taxonomy import (
    BOOK_SOURCES, TIER2_TO_TIER1,
    TIER1_TO_IDX, TIER2_TO_IDX,
    TIER1_LABELS, TIER2_LABELS,
)

# ── Config ─────────────────────────────────────────────────────────────────
STRICT_CHUNK_LIMIT = 2_500   # target chunks per Tier-2 category
CHUNK_SIZE_WORDS   = 100     # fixed words per chunk
REQUEST_SLEEP      = 1.5     # seconds between HTTP requests (avoid IP ban)
REQUEST_TIMEOUT    = 90      # seconds per HTTP call

_GUTEN_MIRRORS = [
    "https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
    "https://gutenberg.pglaf.org/cache/epub/{bid}/pg{bid}.txt",
    "https://gutenberg.org/files/{bid}/{bid}-0.txt",
    "https://gutenberg.org/files/{bid}/{bid}.txt",
]


# ── Network ────────────────────────────────────────────────────────────────

def fetch_book(book_id: int) -> str | None:
    """
    Try each Gutenberg mirror in turn.
    Returns decoded text or None if all mirrors fail.
    Sleeps REQUEST_SLEEP seconds after each attempt to respect rate limits.
    """
    for url_tmpl in _GUTEN_MIRRORS:
        url = url_tmpl.format(bid=book_id)
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            time.sleep(REQUEST_SLEEP)   # ← rate-limit: always sleep after req
            if resp.status_code == 200:
                for enc in ("utf-8", "latin-1", "cp1252"):
                    try:
                        text = resp.content.decode(enc)
                        if len(text) > 500:   # sanity: non-trivial content
                            return text
                    except (UnicodeDecodeError, ValueError):
                        continue
        except requests.RequestException:
            time.sleep(REQUEST_SLEEP)
    return None


# ── Header / Footer Stripping ──────────────────────────────────────────────

# Regex anchors on the canonical Project Gutenberg delimiters
_START_RE = re.compile(r"\*{3}\s*START OF[^\n]*\n", re.IGNORECASE)
_END_RE   = re.compile(r"\*{3}\s*END OF[^\n]*",     re.IGNORECASE)


def strip_gutenberg_boilerplate(raw: str) -> str:
    """
    Remove the PG preamble (license/terms) and trailing footer.
    Falls back to the full text if the delimiters are not found,
    which can happen for some plain-text mirrors.
    """
    start_m = _START_RE.search(raw)
    end_m   = _END_RE.search(raw)
    body    = raw[
        start_m.end() if start_m else 0 :
        end_m.start() if end_m   else len(raw)
    ]
    return body.strip()


# ── Chunking ───────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """
    Fixed-Window Scraper: Read the raw string, clean it, split it into a 
    continuous list of words (tokens), and slice it into fixed chunks of 
    exactly 100 words.
    """
    # Collapse all whitespace and split into words
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    
    chunks: list[str] = []
    
    for i in range(0, len(tokens), CHUNK_SIZE_WORDS):
        chunk = tokens[i:i + CHUNK_SIZE_WORDS]
        if len(chunk) == CHUNK_SIZE_WORDS:
            chunks.append(" ".join(chunk))
            
    return chunks


# ── Audit Table ────────────────────────────────────────────────────────────

def print_audit_table(counts: dict[str, int], final_quota: int) -> None:
    """
    Print chunk counts per Tier-2 category after balancing.
    Shows whether the final_quota equals STRICT_CHUNK_LIMIT or was
    reduced by graceful degradation.
    """
    col = 70
    note = (
        f"[DEGRADED to {final_quota:,}]" if final_quota < STRICT_CHUNK_LIMIT
        else f"[TARGET MET: {final_quota:,}]"
    )
    print("\n" + "═" * col)
    print(f"  PRE-TRAINING DATA AUDIT  {note}")
    print("═" * col)
    hdr = f"{'Tier-2 Category':<22} {'Tier-1 Branch':<18} {'Final Count':>10}  Status"
    print("  " + hdr)
    print("  " + "─" * (col - 2))
    total = 0
    for t2 in TIER2_LABELS:
        n    = counts.get(t2, 0)
        t1   = TIER2_TO_TIER1.get(t2, "?")
        flag = "✓ OK" if n == final_quota else f"⚠ {n}"
        print(f"  {t2:<22} {t1:<18} {n:>10}  {flag}")
        total += n
    print("  " + "─" * (col - 2))
    expected = final_quota * len(TIER2_LABELS)
    ok_str   = "✓" if total == expected else f"⚠ expected {expected:,}"
    print(f"  {'TOTAL':<41} {total:>10}  {ok_str}")
    print("═" * col + "\n")


# ── Master Ingestion ───────────────────────────────────────────────────────

def ingest_all() -> tuple[list[str], list[int], list[int]]:
    """
    Full ingestion pipeline:

    Pass 1 — Scraping
      For each Tier-2 category, cycle through its 10 Gutenberg IDs
      and accumulate chunks until STRICT_CHUNK_LIMIT (2,500) or
      the library is exhausted.

    Pass 2 — Graceful Degradation
      If any category yielded fewer than STRICT_CHUNK_LIMIT chunks,
      compute min_count = min(all category counts) and truncate every
      category pool to min_count.  This enforces perfect parity without
      crashing, discarding the smallest possible number of samples.

    Pass 3 — Audit
      Print the final count table before returning.

    Returns:
        texts     – list of chunk strings
        t1_labels – Tier-1 integer indices
        t2_labels – Tier-2 integer indices
    """

    # ── Pass 1: Scrape ─────────────────────────────────────────────────────
    pools: dict[str, list[str]] = {}   # tier2_label → list of chunks

    print("\n" + "=" * 70)
    print("  PHASE 1 — MULTI-SOURCE DATA INGESTION")
    print(f"  Target: {STRICT_CHUNK_LIMIT:,} chunks × {len(TIER2_LABELS)} categories"
          f" = {STRICT_CHUNK_LIMIT * len(TIER2_LABELS):,} total")
    print("=" * 70)

    cat_bar = tqdm(TIER2_LABELS, desc="  Categories", unit="cat",
                   ncols=75, colour="cyan")

    for t2_name in cat_bar:
        cat_bar.set_postfix_str(t2_name)
        pool: list[str] = []
        book_ids = BOOK_SOURCES.get(t2_name, [])

        book_bar = tqdm(
            book_ids,
            desc=f"    {t2_name[:14]:<14}",
            unit="book",
            leave=False,
            ncols=75,
        )

        for bid in book_bar:
            if len(pool) >= STRICT_CHUNK_LIMIT:
                break

            book_bar.set_postfix_str(f"ID={bid}  pool={len(pool)}")
            raw = fetch_book(bid)

            if raw is None:
                tqdm.write(f"    [✗] ID={bid} — all mirrors failed, skipping.")
                continue

            body   = strip_gutenberg_boilerplate(raw)
            chunks = chunk_text(body)

            still_needed = STRICT_CHUNK_LIMIT - len(pool)
            taken = chunks[:still_needed]
            pool.extend(taken)

            tqdm.write(
                f"    [✓] {t2_name:<16} ID={bid:<6}  "
                f"+{len(taken):>4} chunks  "
                f"({len(pool):>5}/{STRICT_CHUNK_LIMIT})"
            )

        book_bar.close()

        if len(pool) < STRICT_CHUNK_LIMIT:
            tqdm.write(
                f"  [⚠] {t2_name}: exhausted all sources with only "
                f"{len(pool):,} / {STRICT_CHUNK_LIMIT:,} chunks."
            )

        pools[t2_name] = pool

    cat_bar.close()

    # ── Pass 2: Graceful Degradation ───────────────────────────────────────
    #
    # If any category is short, compute the lowest common denominator:
    #   min_count = min(len(pool) for each category)
    # Then truncate every pool to min_count.
    # This guarantees:
    #   • Perfect mathematical parity (equal class counts).
    #   • No crashes or padding with synthetic data.
    #   • Minimal information loss (only excess chunks from
    #     well-sourced categories are discarded).
    #
    min_count = min(len(pool) for pool in pools.values())

    if min_count < STRICT_CHUNK_LIMIT:
        print(
            f"\n  [Graceful Degradation] Minimum across all categories: "
            f"{min_count:,}  (target was {STRICT_CHUNK_LIMIT:,})."
        )
        print(
            f"  Down-sampling all categories to {min_count:,} chunks "
            f"to ensure perfect parity."
        )
        for t2_name in pools:
            pools[t2_name] = pools[t2_name][:min_count]

    final_quota = min_count   # actual balanced count per category

    # ── Pass 3: Build output arrays ────────────────────────────────────────
    texts:     list[str] = []
    t1_labels: list[int] = []
    t2_labels: list[int] = []

    for t2_name, pool in pools.items():
        t1_name = TIER2_TO_TIER1[t2_name]
        t1_idx  = TIER1_TO_IDX[t1_name]
        t2_idx  = TIER2_TO_IDX[t2_name]
        texts.extend(pool)
        t1_labels.extend([t1_idx] * len(pool))
        t2_labels.extend([t2_idx] * len(pool))

    # ── Audit printout ─────────────────────────────────────────────────────
    actual_counts = {t2: len(p) for t2, p in pools.items()}
    print_audit_table(actual_counts, final_quota)
    print(f"  Total samples passed to vectorizer: {len(texts):,}")

    return texts, t1_labels, t2_labels
