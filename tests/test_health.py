from unittest.mock import patch


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "checks" in body


def test_health_structure(client):
    response = client.get("/health")
    body = response.json()
    assert "status" in body
    assert "secret_key" in body["checks"]
    assert "artifacts" in body["checks"]


def test_health_no_models_dir(client):
    with patch("src.api.routers.health._MODELS_DIR") as mock_dir:
        mock_dir.exists.return_value = False
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Ford VinGuard ML API",
        "docs": "/docs",
        "readiness": "/health",
    }
