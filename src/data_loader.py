import requests, os, time, re
from prefect import task, flow, get_run_logger

BASE_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

BOOK_IDS = {
    "Existentialism": [5827, 4280],
    "Rationalism":    [59,   4705],
    "Empiricism":     [9662, 10777],
    "Stoicism":       [2680, 4582],
    "Ethics":         [8438, 6433],
    "Political":      [1232, 3207],
}

@task(retries=3, retry_delay_seconds=5)
def fetch_book(book_id: int, label: str) -> dict:
    logger = get_run_logger()
    url = BASE_URL.format(id=book_id)
    logger.info(f"Fetching book {book_id} — {label}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return {"text": response.text, "label": label, "id": book_id}

@task
def clean_gutenberg_text(raw: dict) -> dict:
    """Remove Gutenberg header/footer boilerplate"""
    text = raw["text"]
    # Strip everything before "START OF" and after "END OF"
    start = re.search(r"\*\*\* START OF .+? \*\*\*", text)
    end   = re.search(r"\*\*\* END OF .+? \*\*\*",   text)
    if start and end:
        text = text[start.end():end.start()]
    raw["text"] = text.strip()
    return raw

@task
def save_text(data: dict, out_dir: str = "data") -> str:
    os.makedirs(f"{out_dir}/{data['label']}", exist_ok=True)
    path = f"{out_dir}/{data['label']}/book_{data['id']}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(data["text"])
    return path

@flow(name="Ingest Philosophy Texts")
def ingest_flow():
    for label, ids in BOOK_IDS.items():
        for book_id in ids:
            raw     = fetch_book(book_id, label)
            cleaned = clean_gutenberg_text(raw)
            path    = save_text(cleaned)
            time.sleep(1)  # Be polite to Gutenberg's servers

if __name__ == "__main__":
    ingest_flow()