"""Shared test fixtures for Philosophical Engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


# Sample text for testing
SAMPLE_TEXT = """
Philosophy is the study of general and fundamental questions about existence,
knowledge, values, reason, mind, and language. It is a fundamental activity of
human thought. Philosophers try to understand the nature of reality and our place
in the universe.
"""

SAMPLE_TEXTS = [
    """
    Empiricism is the view that all knowledge is derived from sense-experience.
    It emphasizes the role of empirical evidence in the formation of ideas.
    John Locke and David Hume are famous empiricist philosophers.
    """,
    """
    Rationalism emphasizes reason as the primary source of knowledge.
    René Descartes believed that reason alone could yield certain knowledge.
    Rationalists claim that certain truths can be known through reason alone.
    """,
    """
    Existentialism focuses on individual existence, freedom, and choice.
    Jean-Paul Sartre and Friedrich Nietzsche explored these themes.
    It emphasizes personal responsibility and authentic living.
    """,
]


@pytest.fixture
def sample_text() -> str:
    """Return a sample philosophical text for testing."""
    return SAMPLE_TEXT


@pytest.fixture
def sample_texts() -> list[str]:
    """Return a list of sample philosophical texts."""
    return SAMPLE_TEXTS


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample DataFrame with processed book features."""
    data: list[dict[str, Any]] = []
    for i, text in enumerate(SAMPLE_TEXTS):
        data.append(
            {
                "gutenberg_id": 1000 + i,
                "title": f"Philosophical Work {i+1}",
                "author": f"Philosopher {i+1}",
                "year": 1800 + i * 50,
                "subjects": ["Philosophy"],
                "download_count": 1000 * (i + 1),
                "avg_sentence_length": 15.0 + i,
                "vocab_richness": 0.5 + i * 0.05,
                "sentiment_polarity": 0.1 * (i - 1),
                "named_entity_count": 3 + i,
                "top_concepts": ["knowledge", "reason", "philosophy"],
                "decade": 1800 + i * 50,
                "era_label": "Enlightenment" if i == 0 else "Modern",
                "school_label": "Empiricism" if i == 0 else "Rationalism",
                "full_text": text,
            }
        )
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_csv(tmp_path: Path) -> str:
    """Create a temporary CSV file with sample data and return its path."""
    df = pd.DataFrame(
        {
            "gutenberg_id": [1001, 1002, 1003],
            "title": ["Book A", "Book B", "Book C"],
            "author": ["Author A", "Author B", "Author C"],
            "year": [1850, 1900, 1950],
            "subjects": [["Philosophy"]] * 3,
            "download_count": [1000, 2000, 500],
            "avg_sentence_length": [12.0, 15.0, 18.0],
            "vocab_richness": [0.55, 0.60, 0.65],
            "sentiment_polarity": [0.1, 0.2, -0.1],
            "named_entity_count": [5, 8, 3],
            "top_concepts": [
                ["reason", "truth"],
                ["existence", "freedom"],
                ["knowledge", "experience"],
            ],
            "decade": [1850, 1900, 1950],
            "era_label": ["Modern", "Modern", "Contemporary"],
            "school_label": ["Rationalism", "Existentialism", "Empiricism"],
            "full_text": SAMPLE_TEXTS,
        }
    )
    csv_path = tmp_path / "test_corpus.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_raw_json(tmp_path: Path) -> Path:
    """Create a temporary raw JSON file and return its path."""
    data = {
        "gutenberg_id": 12345,
        "title": "Test Philosophy",
        "author": "Test Author",
        "year": 1900,
        "subjects": ["Philosophy"],
        "download_count": 500,
        "avg_sentence_length": 15.0,
        "vocab_richness": 0.6,
        "sentiment_polarity": 0.2,
        "named_entity_count": 5,
        "top_concepts": ["reason", "truth"],
        "decade": 1900,
        "era_label": "Modern",
        "school_label": "Rationalism",
        "full_text": SAMPLE_TEXT,
    }
    json_path = tmp_path / "12345.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return json_path


@pytest.fixture
def loaded_models() -> dict[str, Any]:
    """Return a dictionary simulating loaded model artifacts."""
    # Phase 1: no actual models loaded
    return {}


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory."""
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    return models
