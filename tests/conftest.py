import os

os.environ.setdefault("SECRET_KEY", "x" * 32)
os.environ.setdefault("ML_SERVICE_TOKEN", "test-service-token-fixed")

import pytest


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def analyst_token():
    from src.api.security.auth import create_access_token

    return create_access_token(subject="test-analyst", role="analyst")


@pytest.fixture(scope="session")
def viewer_token():
    from src.api.security.auth import create_access_token

    return create_access_token(subject="test-viewer", role="viewer")


@pytest.fixture(scope="session")
def admin_token():
    from src.api.security.auth import create_access_token

    return create_access_token(subject="test-admin", role="admin")


@pytest.fixture(scope="session")
def analyst_headers(analyst_token):
    return {"Authorization": f"Bearer {analyst_token}"}


@pytest.fixture(scope="session")
def viewer_headers(viewer_token):
    return {"Authorization": f"Bearer {viewer_token}"}


@pytest.fixture(scope="session")
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="session")
def service_token_headers():
    return {"X-ML-Service-Token": "test-service-token-fixed"}
