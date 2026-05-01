"""Text preprocessing utilities: tokenization and lemmatization."""

from __future__ import annotations

import re
from typing import List

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
)

_lemmatizer = WordNetLemmatizer()


def ensure_nltk_resources() -> None:
    """Download required NLTK resources when missing."""
    for lookup_path, resource_name in _NLTK_RESOURCES:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for consistent processing."""
    lowered = text.lower()
    return re.sub(r"\s+", " ", lowered).strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text into word-like units."""
    ensure_nltk_resources()
    normalized = normalize_text(text)
    tokens = word_tokenize(normalized)
    return [token for token in tokens if re.search(r"[a-z0-9]", token)]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize token list using NLTK WordNet lemmatizer."""
    ensure_nltk_resources()
    return [_lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(text: str) -> List[str]:
    """Run the full preprocessing pipeline (tokenization + lemmatization)."""
    tokens = tokenize(text)
    return lemmatize_tokens(tokens)
