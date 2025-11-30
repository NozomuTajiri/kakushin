"""Tests for health check API endpoints."""

import pytest
from fastapi.testclient import TestClient

from kakushin.api.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


def test_health_check(client):
    """Test basic health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
    assert "gpu_available" in data


def test_readiness_check(client):
    """Test readiness probe endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_liveness_check(client):
    """Test liveness probe endpoint."""
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_gpu_info(client):
    """Test GPU information endpoint."""
    response = client.get("/health/gpu")
    assert response.status_code == 200

    data = response.json()
    assert "available" in data
    assert "device_count" in data
    assert "devices" in data
    assert isinstance(data["devices"], list)
