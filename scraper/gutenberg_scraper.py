import os
import re
import json
import time
import math
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin

import spacy
import requests
import pandas as pd
from textblob import TextBlob
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Attempt to load spacy model, download if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.gutenberg.org"
SEARCH_URL = "https://www.gutenberg.org/ebooks/search/"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class ScraperConfig:
    max_books: int = 500
    rate_limit_sec: float = 2.0
    max_retries: int = 3
    query: str = "philosophy"

def get_session():
    session = requests.Session()
    retries = Retry(total=ScraperConfig.max_retries, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

def clean_text(text: str) -> str:
    """Basic clean up for raw Gutenberg text."""
    start_match = re.search(r"\*\*\* START OF [^\n]*\*\*\*", text)
    end_match = re.search(r"\*\*\* END OF [^\n]*\*\*\*", text)
    if start_match and end_match:
        text = text[start_match.end():end_match.start()]
    return text.strip()

def map_school_label(subjects: List[str]) -> str:
    subjects_lower = " ".join(subjects).lower()
    if "empiricism" in subjects_lower: return "Empiricism"
    if "rationalism" in subjects_lower: return "Rationalism"
    if "existentialism" in subjects_lower: return "Existentialism"
    if "stoic" in subjects_lower: return "Stoicism"
    if "idealism" in subjects_lower: return "Idealism"
    if "pragmatism" in subjects_lower: return "Pragmatism"
    return "Other"

def determine_era(year: int) -> str:
    if year < 500: return "Ancient"
    elif year < 1400: return "Medieval"
    elif year < 1600: return "Renaissance"
    elif year < 1800: return "Enlightenment"
    elif year < 1950: return "Modern"
    else: return "Contemporary"

def extract_features(text: str) -> Dict[str, Any]:
    doc = nlp(text[:100000])  # process at most first 100k chars for performance
    
    sentences = list(doc.sents)
    avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
    
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    vocab_richness = len(set(tokens)) / len(tokens) if tokens else 0
    
    blob = TextBlob(text[:100000])
    sentiment_polarity = blob.sentiment.polarity
    
    named_entity_count = len(doc.ents)
    
    nouns = [t.lemma_.lower() for t in doc if t.pos_ == "NOUN" and not t.is_stop]
    top_concepts = [item[0] for item in Counter(nouns).most_common(10)]
    
    return {
        "avg_sentence_length": round(avg_sentence_length, 2),
        "vocab_richness": round(vocab_richness, 4),
        "sentiment_polarity": round(sentiment_polarity, 4),
        "named_entity_count": named_entity_count,
        "top_concepts": top_concepts
    }

def process_book_metadata(session: requests.Session, book_link: str) -> Optional[Dict[str, Any]]:
    book_url = urljoin(BASE_URL, book_link)
    book_id_match = re.search(r"/ebooks/(\d+)", book_link)
    if not book_id_match:
        return None
    
    book_id = book_id_match.group(1)
    
    raw_file = RAW_DIR / f"{book_id}.json"
    if raw_file.exists():
        logger.info(f"Book {book_id} already cached.")
        with open(raw_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    time.sleep(ScraperConfig.rate_limit_sec)
    
    try:
        res = session.get(book_url, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch metadata for {book_id}: {e}")
        return None
        
    soup = BeautifulSoup(res.text, "html.parser")
    
    metadata = {
        "gutenberg_id": book_id,
        "title": "Unknown",
        "author": "Unknown",
        "year": 1900,
        "subjects": [],
        "download_count": 0,
        "full_text": ""
    }
    
    table = soup.find("table", {"class": "bibrec"})
    if table:
        for row in table.find_all("tr"):
            th = row.find("th")
            td = row.find("td")
            if not th or not td: continue
            
            key = th.text.strip().lower()
            if "title" in key:
                metadata["title"] = td.text.strip().replace("\n", " ")
            elif "author" in key:
                metadata["author"] = td.text.strip().split("\n")[0]
            elif "subject" in key:
                metadata["subjects"].extend([a.text for a in td.find_all("a")])
                
    year_match = re.search(r"(\d{4})-\d{4}", metadata["author"])
    if year_match:
        metadata["year"] = int(year_match.group(1))
        
    downloads_elem = soup.find("td", {"itemprop": "downloads"})
    if downloads_elem:
        dl_text = downloads_elem.text.strip().split()[0]
        metadata["download_count"] = int(dl_text) if dl_text.isdigit() else 0
        
    txt_url = None
    for a in soup.find_all("a", class_="link"):
        if "Plain Text UTF-8" in a.text or "text/plain" in getattr(a, "attrs", {}).get("type", ""):
            txt_url = urljoin(BASE_URL, a["href"])
            break
            
    if not txt_url:
        logger.warning(f"No plain text found for {book_id}")
        return None
        
    time.sleep(ScraperConfig.rate_limit_sec)
    try:
        txt_res = session.get(txt_url, timeout=20)
        txt_res.encoding = 'utf-8'
        metadata["full_text"] = clean_text(txt_res.text)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch text for {book_id}: {e}")
        return None
        
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    return metadata

def scrape_gutenberg(max_books: int = ScraperConfig.max_books) -> pd.DataFrame:
    session = get_session()
    start_index = 0
    books_collected = 0
    all_processed_data = []
    
    logger.info("Starting Gutenberg scrape...")
    
    while books_collected < max_books:
        url = f"{SEARCH_URL}?query={ScraperConfig.query}&submit_search=Search&start_index={start_index}"
        try:
            res = session.get(url, timeout=10)
            res.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Search page failed: {e}")
            break
            
        soup = BeautifulSoup(res.text, "html.parser")
        book_links = [a["href"] for a in soup.select(".booklink a.link")]
        
        if not book_links:
            logger.info("No more books found.")
            break
            
        for link in book_links:
            if books_collected >= max_books:
                break
                
            raw_meta = process_book_metadata(session, link)
            if not raw_meta or not raw_meta.get("full_text"):
                continue
                
            logger.info(f"Processing book {books_collected + 1}/{max_books}: {raw_meta['title']}")
            
            features = extract_features(raw_meta["full_text"])
            
            year = raw_meta["year"]
            decade = round(year / 10) * 10
            
            row = {
                "gutenberg_id": raw_meta["gutenberg_id"],
                "title": raw_meta["title"],
                "author": raw_meta["author"],
                "year": year,
                "decade": decade,
                "era_label": determine_era(year),
                "school_label": map_school_label(raw_meta["subjects"]),
                "subjects": "|".join(raw_meta["subjects"]),
                "download_count": raw_meta["download_count"],
                "text_length": len(raw_meta["full_text"]),
                **features
            }
            
            all_processed_data.append(row)
            books_collected += 1
            
        start_index += 25
        
    if all_processed_data:
        df = pd.DataFrame(all_processed_data)
        out_path = PROCESSED_DIR / "philosophy_corpus.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(df)} records to {out_path}")
        return df
    else:
        logger.warning("No books processed.")
        return pd.DataFrame()

if __name__ == "__main__":
    scrape_gutenberg(max_books=10)
