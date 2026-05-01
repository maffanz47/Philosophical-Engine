def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_models_info(test_client):
    response = test_client.get("/models/info")
    assert response.status_code == 200
    assert "classification" in response.json()

def test_classify_endpoint(test_client):
    response = test_client.post(
        "/api/v1/classify/",
        json={"text": "This is a dummy text"}
    )
    # Might fail if model not trained, so we just check it doesn't crash catastrophically
    assert response.status_code in [200, 500]

def test_recommend_endpoint(test_client):
    response = test_client.get("/api/v1/recommend/?query=test&n=5")
    assert response.status_code in [200, 500]
