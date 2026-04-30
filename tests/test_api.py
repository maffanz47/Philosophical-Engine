import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from contextlib import asynccontextmanager
import numpy as np
import torch
import main

# Mock the lifespan context manager so the app doesn't try to load files from disk
@asynccontextmanager
async def mock_lifespan(app):
    main.state["w2v"] = MagicMock()
    
    # Mock model to return a dummy logits tensor for 1 sample and 6 classes
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(1, 6)
    main.state["model"] = mock_model
    
    mock_sklearn = MagicMock()
    mock_sklearn.predict_proba.return_value = np.random.rand(1, 6)
    main.state["logreg"] = mock_sklearn
    main.state["rf"] = mock_sklearn
    
    main.state["all_vecs"] = np.random.rand(10, 200)
    main.state["metadata"] = [{"label": "Ethics", "text": "Preview text..."} for _ in range(10)]
    yield

main.app.router.lifespan_context = mock_lifespan

@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c

def test_predict_endpoint(client):
    response = client.post("/predict", json={"text": "I think therefore I am."})
    assert response.status_code == 200
    data = response.json()
    assert "consensus_theme" in data
    assert "ensemble_results" in data
    assert "suggested_reading" in data
    assert "Neural_Network" in data["ensemble_results"]
    assert "Logistic_Regression" in data["ensemble_results"]
    assert "Random_Forest" in data["ensemble_results"]
    assert "dominant_theme" in data["ensemble_results"]["Neural_Network"]
    
def test_similar_endpoint(client):
    response = client.get("/similar?text=What is the nature of reality&top_k=2")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert "source" in data["results"][0]
    assert "score" in data["results"][0]
