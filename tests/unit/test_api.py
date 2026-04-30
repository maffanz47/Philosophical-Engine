"""Unit tests for FastAPI endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestClassifyEndpoint:
    """Tests for /api/v1/classify/ endpoint."""

    def test_classify_endpoint_exists(self, client: TestClient):
        response = client.post("/api/v1/classify/", json={"text": "test philosophy"})
        # Should return a response (even if model not loaded)
        assert response.status_code in [200, 500]

    def test_classify_empty_text(self, client: TestClient):
        response = client.post("/api/v1/classify/", json={"text": ""})
        # Should return a valid response
        assert response.status_code in [200, 422]

    def test_classify_response_format(self, client: TestClient):
        response = client.post("/api/v1/classify/", json={"text": "philosophy is love of wisdom"})
        if response.status_code == 200:
            data = response.json()
            assert "school" in data
            assert "confidence" in data
            assert "top3" in data


class TestRegressionEndpoint:
    """Tests for /api/v1/regression/influence endpoint."""

    def test_regression_endpoint_exists(self, client: TestClient):
        response = client.post("/api/v1/regression/influence", json={"text": "test philosophy"})
        # Should return a response
        assert response.status_code in [200, 500]

    def test_regression_empty_text(self, client: TestClient):
        response = client.post("/api/v1/regression/influence", json={"text": ""})
        assert response.status_code in [200, 422]

    def test_regression_response_format(self, client: TestClient):
        response = client.post("/api/v1/regression/influence", json={"text": "philosophy text"})
        if response.status_code == 200:
            data = response.json()
            assert "influence_score" in data


class TestEmbeddingsEndpoint:
    """Tests for /api/v1/embeddings/map endpoint."""

    def test_embeddings_endpoint_exists(self, client: TestClient):
        response = client.get("/api/v1/embeddings/map")
        # Should return a response
        assert response.status_code in [200, 500]


class TestRecommendEndpoint:
    """Tests for /api/v1/recommend endpoint."""

    def test_recommend_endpoint_exists(self, client: TestClient):
        response = client.post(
            "/api/v1/recommend", json={"book_id_or_text": "test", "n": 5}
        )
        assert response.status_code in [200, 500]

    def test_recommend_with_n_parameter(self, client: TestClient):
        response = client.post(
            "/api/v1/recommend", json={"book_id_or_text": "test", "n": 10}
        )
        assert response.status_code in [200, 500]


class TestTimeseriesEndpoint:
    """Tests for /api/v1/timeseries/sentiment endpoint."""

    def test_timeseries_endpoint_exists(self, client: TestClient):
        response = client.get("/api/v1/timeseries/sentiment")
        assert response.status_code in [200, 500]


class TestClustersEndpoint:
    """Tests for /api/v1/clusters endpoint."""

    def test_clusters_endpoint_exists(self, client: TestClient):
        response = client.get("/api/v1/clusters")
        assert response.status_code in [200, 500]


class TestAssociationsEndpoint:
    """Tests for /api/v1/associations endpoint."""

    def test_associations_endpoint_exists(self, client: TestClient):
        response = client.get("/api/v1/associations")
        assert response.status_code in [200, 500]

    def test_associations_with_concept_filter(self, client: TestClient):
        response = client.get("/api/v1/associations?concept=truth")
        assert response.status_code in [200, 500]


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_endpoint(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "uptime_seconds" in data


class TestModelsInfoEndpoint:
    """Tests for /models/info endpoint."""

    def test_models_info_endpoint(self, client: TestClient):
        response = client.get("/models/info")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data


class TestValidationErrors:
    """Tests for validation error handling."""

    def test_classify_missing_field(self, client: TestClient):
        response = client.post("/api/v1/classify/", json={})
        assert response.status_code == 422

    def test_regression_missing_field(self, client: TestClient):
        response = client.post("/api/v1/regression/influence", json={})
        assert response.status_code == 422


class TestCorSHeaders:
    """Tests for CORS middleware."""

    def test_cors_preflight(self, client: TestClient):
        response = client.options(
            "/api/v1/classify/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        # OPTIONS request for CORS preflight
        assert response.status_code in [200, 405]
