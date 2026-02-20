"""
Integration tests for the FastAPI endpoints.
Uses a mock model so no training is required.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Patch ModelStore before importing app
@pytest.fixture(scope="module")
def client():
    mock_store = MagicMock()
    mock_store.is_ready = True
    mock_store.threshold = 0.3
    mock_store.model_name = "xgboost"
    mock_store.predict_proba.return_value = 0.15  # below threshold → no failure

    with patch("src.api.model_store.ModelStore.get", return_value=mock_store):
        from src.api.app import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c


VALID_READING = {
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100,
}


def test_health_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_no_failure(client):
    resp = client.post("/api/v1/predict", json=VALID_READING)
    assert resp.status_code == 200
    data = resp.json()
    assert "failure_predicted" in data
    assert "failure_probability" in data
    assert data["failure_probability"] == 0.15
    assert data["failure_predicted"] is False


def test_predict_invalid_rpm(client):
    bad = {**VALID_READING, "Rotational speed [rpm]": 50}  # below min=1000
    resp = client.post("/api/v1/predict", json=bad)
    assert resp.status_code == 422


def test_predict_invalid_type(client):
    bad = {**VALID_READING, "Type": "X"}
    resp = client.post("/api/v1/predict", json=bad)
    assert resp.status_code == 422
