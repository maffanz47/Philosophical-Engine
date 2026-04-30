import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def sample_text():
    return "This is a sample philosophical text about the nature of being and existence."

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "gutenberg_id": [1, 2],
        "title": ["Critique of Pure Reason", "Beyond Good and Evil"],
        "author": ["Immanuel Kant", "Friedrich Nietzsche"],
        "year": [1781, 1886],
        "school_label": ["Idealism", "Existentialism"],
        "full_text": ["Sample Kant text.", "Sample Nietzsche text."],
        "avg_sentence_length": [20.5, 15.2]
    })

@pytest.fixture
def test_client():
    return TestClient(app)
