"""
Integration Tests for FastAPI Endpoints
Tests health, readiness probe (503 vs 200), models-info, samples, and predict endpoints.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.app import app

@pytest.fixture(scope="module")
def client():
    # Context manager triggers the FastAPI lifespan startup event
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint(client):
    """Verify GET /api/health returns 200 and ok status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "deepfake-audio-detector"

def test_readiness_endpoint(client):
    """Verify GET /api/ready returns 200 and models_loaded=3."""
    res = client.get("/api/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["models_loaded"] == 3
    assert data["scaler_loaded"] is True

def test_models_info_endpoint(client):
    """Verify GET /api/models-info returns 3 models and 26 features."""
    res = client.get("/api/models-info")
    assert res.status_code == 200
    data = res.json()
    assert len(data["models"]) == 3
    assert data["features_count"] == 26
    assert len(data["features_list"]) == 26

def test_samples_endpoint(client):
    """Verify GET /api/samples returns list of available test audios."""
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert len(data["samples"]) > 0

def test_predict_with_sample_id(client):
    """Verify POST /api/predict correctly analyzes a sample ID."""
    res = client.post("/api/predict", data={"sample_id": "sample_real_1"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["final_decision"] in ["REAL", "FAKE"]
    assert "majority_vote" in data
    assert "models" in data
    assert "features" in data
    assert len(data["features"]) == 26
    assert "processing_time_ms" in data

def test_predict_with_invalid_sample_id(client):
    """Verify POST /api/predict returns 404 for unknown sample ID."""
    res = client.post("/api/predict", data={"sample_id": "non_existent_sample_xyz"})
    assert res.status_code == 404

def test_predict_without_payload(client):
    """Verify POST /api/predict returns 400 when neither file nor sample_id is passed."""
    res = client.post("/api/predict")
    assert res.status_code == 400

if __name__ == "__main__":
    pytest.main(["-v", __file__])

