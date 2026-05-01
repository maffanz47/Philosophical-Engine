"""
test_api.py
===========
Unit tests for preprocessing functions and API endpoint responsiveness.
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend" / "src"))
sys.path.insert(0, str(BASE_DIR / "backend"))

from preprocessor import flesch_kincaid_grade, avg_sentence_length, lexical_diversity, compute_complexity_score
from main import app

client = TestClient(app)

# --- Preprocessor Unit Tests ---

def test_flesch_kincaid_grade():
    text = "The quick brown fox jumps over the lazy dog. It is a very simple sentence."
    score = flesch_kincaid_grade(text)
    assert isinstance(score, float)
    assert 0 <= score <= 20

def test_avg_sentence_length():
    text = "Hello world! How are you today? I am fine."
    avg_len = avg_sentence_length(text)
    assert avg_len > 0
    assert avg_len == (2 + 4 + 3) / 3

def test_lexical_diversity():
    text = "apple orange apple banana"
    ttr = lexical_diversity(text)
    assert ttr == 3 / 4

def test_compute_complexity_score():
    text = "This is a philosophical text that ponders the existence of meaning in a seemingly absurd universe."
    score = compute_complexity_score(text)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

# --- API Endpoint Tests ---

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_schools_endpoint():
    response = client.get("/schools")
    assert response.status_code == 200
    data = response.json()
    assert "schools" in data
    assert len(data["schools"]) > 0

def test_predict_endpoint_short_text():
    response = client.post("/predict", json={"text": "too short", "model_tier": "baseline"})
    assert response.status_code == 422
