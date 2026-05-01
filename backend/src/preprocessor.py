"""
preprocessor.py
===============
Text preprocessing pipeline:
  1. Strip Gutenberg boilerplate
  2. Tokenization & Lemmatization (spaCy)
  3. Chunk text into 300-500 word segments
  4. Compute Reading Complexity Score:
       - Flesch-Kincaid Grade Level
       - Average Sentence Length
       - Lexical Diversity (Type-Token Ratio)
  5. Build corpus DataFrame
"""

import re
import logging
import unicodedata
from pathlib import Path
from typing import Optional

import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("preprocessor")

# Load spaCy once
try:
    NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    NLP.add_pipe("sentencizer")
except OSError:
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    NLP = None

CHUNK_MIN = 300
CHUNK_MAX = 500
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

_START_MARKERS = [
    "*** START OF THIS PROJECT GUTENBERG",
    "*** START OF THE PROJECT GUTENBERG",
]
_END_MARKERS = [
    "*** END OF THIS PROJECT GUTENBERG",
    "*** END OF THE PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "End of Project Gutenberg",
]


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate."""
    text = unicodedata.normalize("NFKD", raw_text)
    start_idx = 0
    for marker in _START_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            start_idx = text.find("\n", idx) + 1
            break
    end_idx = len(text)
    for marker in _END_MARKERS:
        idx = text.find(marker, start_idx)
        if idx != -1:
            end_idx = idx
            break
    body = text[start_idx:end_idx]
    body = re.sub(r"\r\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def tokenize_and_lemmatize(text: str) -> list:
    """Tokenize and lemmatize text using spaCy."""
    if NLP is None:
        raise RuntimeError("spaCy not loaded. Run: python -m spacy download en_core_web_sm")
    doc = NLP(text[:1_000_000])
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
        and not token.is_space and len(token.text) > 2
    ]


def chunk_text(text: str, min_words: int = CHUNK_MIN, max_words: int = CHUNK_MAX) -> list:
    """Split cleaned text into 300-500 word chunks with 50-word overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_words = []
    for sentence in sentences:
        words = sentence.split()
        current_words.extend(words)
        if len(current_words) >= max_words:
            chunk = " ".join(current_words[:max_words])
            chunks.append(chunk)
            current_words = current_words[max_words - 50:]
    if len(current_words) >= min_words:
        chunks.append(" ".join(current_words))
    return chunks


def _count_syllables(word: str) -> int:
    """Rough syllable count via vowel-group heuristics."""
    word = word.lower().strip(".,!?;:")
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def flesch_kincaid_grade(text: str) -> float:
    """Compute Flesch-Kincaid Grade Level, clamped to [0, 20]."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = text.split()
    n_sentences = max(len(sentences), 1)
    n_words = max(len(words), 1)
    n_syllables = sum(_count_syllables(w) for w in words)
    fk = 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59
    return float(np.clip(fk, 0, 20))


def avg_sentence_length(text: str) -> float:
    """Average words per sentence."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0
    return float(np.mean([len(s.split()) for s in sentences]))

def avg_word_length(text: str) -> float:
    """Average characters per word."""
    words = text.split()
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))

def lexical_diversity(text: str) -> float:
    """Type-Token Ratio (unique_words / total_words) in [0, 1]."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_complexity_score(text: str) -> float:
    """
    Composite Reading Complexity Score in [0, 1].

    Weighted formula:
        score = 0.40 * (fk/20) + 0.35 * (asl/50) + 0.25 * ttr
    """
    fk  = flesch_kincaid_grade(text)
    asl = avg_sentence_length(text)
    ttr = lexical_diversity(text)
    score = 0.40 * float(np.clip(fk / 20.0, 0, 1)) \
          + 0.35 * float(np.clip(asl / 50.0, 0, 1)) \
          + 0.25 * ttr
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def balance_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Stratified Sampling with oversampling to fix class imbalance."""
    if df.empty:
        return df
    max_count = df['school'].value_counts().max()
    balanced = []
    for school, group in df.groupby('school'):
        if len(group) < max_count:
            group = group.sample(max_count, replace=True, random_state=42)
        balanced.append(group)
    balanced_df = pd.concat(balanced).reset_index(drop=True)
    logger.info("Corpus balanced. New class distribution:\n%s", balanced_df['school'].value_counts())
    return balanced_df


def build_corpus(manifest: list, save_csv: bool = True) -> pd.DataFrame:
    """
    Process every book in the manifest into a chunked, labeled DataFrame.

    Columns: chunk_id, book_id, school, author, title,
             raw_chunk, lemmas, complexity_score, fk_grade,
             avg_sent_len, avg_word_len, lexical_div
    """
    rows = []
    chunk_counter = 0

    for entry in tqdm(manifest, desc="Processing books"):
        book_id = entry["book_id"]
        path = Path(entry["path"])
        if not path.exists():
            logger.warning("File not found: %s — skipping.", path)
            continue

        logger.info("Processing '%s' by %s …", entry["title"], entry["author"])
        raw_text = path.read_text(encoding="utf-8", errors="replace")
        cleaned  = strip_gutenberg_boilerplate(raw_text)
        chunks   = chunk_text(cleaned)
        logger.info("  → %d chunks generated.", len(chunks))

        for chunk in chunks:
            try:
                lemmas = tokenize_and_lemmatize(chunk)
                lemma_str = " ".join(lemmas)
            except RuntimeError:
                lemma_str = chunk.lower()

            rows.append({
                "chunk_id":         chunk_counter,
                "book_id":          book_id,
                "school":           entry["school"],
                "author":           entry["author"],
                "title":            entry["title"],
                "raw_chunk":        chunk,
                "lemmas":           lemma_str,
                "complexity_score": compute_complexity_score(chunk),
                "fk_grade":         round(flesch_kincaid_grade(chunk), 4),
                "avg_sent_len":     round(avg_sentence_length(chunk), 4),
                "avg_word_len":     round(avg_word_length(chunk), 4),
                "lexical_div":      round(lexical_diversity(chunk), 4),
            })
            chunk_counter += 1

    df = pd.DataFrame(rows)
    df = balance_corpus(df)
    
    logger.info("Corpus built: %d chunks.", len(df))

    if save_csv and not df.empty:
        out_path = PROCESSED_DIR / "corpus.csv"
        df.to_csv(out_path, index=False)
        logger.info("Corpus saved → %s", out_path)

    return df


if __name__ == "__main__":
    import json
    manifest_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "manifest.json"
    if not manifest_path.exists():
        print("No manifest found. Run data_loader.py first.")
    else:
        manifest = json.loads(manifest_path.read_text())
        df = build_corpus(manifest)
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"Schools:\n{df['school'].value_counts()}")
