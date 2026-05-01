from io import BytesIO

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_predict_with_json_text() -> None:
    payload = {"text": "Virtue is achieved through disciplined reason and ethics."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_school" in data
    assert "confidence_score" in data
    assert "complexity_index" in data
    assert "top_3_recommendations" in data
    assert len(data["top_3_recommendations"]) == 3


def test_predict_with_txt_upload() -> None:
    file_content = b"Meaning emerges from freedom, anguish, and personal choice."
    files = {"file": ("input.txt", BytesIO(file_content), "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    assert "predicted_school" in response.json()
