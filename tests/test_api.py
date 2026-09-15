"""
Integration tests for the Flask Web & REST API endpoints.
"""

import pytest
import json
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test GET /api/health."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_home_page(client):
    """Test GET / renders dashboard HTML."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"CardioSense" in res.data
    assert b"Patient Clinical Biomarkers" in res.data


def test_models_page(client):
    """Test GET /models renders benchmark and audit HTML."""
    res = client.get("/models")
    assert res.status_code == 200
    assert b"Model Benchmarks" in res.data
    assert b"Technical Autopsy" in res.data


def test_predict_api_valid(client):
    """Test POST /api/predict with valid payload."""
    payload = {
        "Age": 55,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 135,
        "Cholesterol": 245,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 130,
        "ExerciseAngina": "N",
        "Oldpeak": 1.2,
        "ST_Slope": "Flat"
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["success"] is True
    assert "probability_percent" in data["data"]
    assert "risk_tier" in data["data"]
    assert "clinical_summary" in data["data"]


def test_predict_api_invalid_range(client):
    """Test POST /api/predict with out-of-range data returns 422."""
    payload = {
        "Age": 250,  # Invalid age
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 135,
        "Cholesterol": 245,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 130,
        "ExerciseAngina": "N",
        "Oldpeak": 1.2,
        "ST_Slope": "Flat"
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 422
    data = json.loads(res.data)
    assert data["success"] is False
    assert len(data["errors"]) > 0


def test_predict_api_missing_fields(client):
    """Test POST /api/predict with missing fields returns 422."""
    payload = {"Age": 55}
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 422
    data = json.loads(res.data)
    assert data["success"] is False
    assert len(data["errors"]) > 0


def test_models_api(client):
    """Test GET /api/models returns benchmark data."""
    res = client.get("/api/models")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["success"] is True
    assert "benchmarks" in data
    assert "Random Forest" in data["benchmarks"]


def test_batch_predict_json(client):
    """Test POST /api/batch-predict with JSON array."""
    patients = [
        {
            "Age": 32, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 110,
            "Cholesterol": 170, "FastingBS": 0, "RestingECG": "Normal",
            "MaxHR": 180, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up"
        },
        {
            "Age": 65, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 160,
            "Cholesterol": 280, "FastingBS": 1, "RestingECG": "ST",
            "MaxHR": 110, "ExerciseAngina": "Y", "Oldpeak": 2.5, "ST_Slope": "Flat"
        }
    ]
    res = client.post("/api/batch-predict", json=patients)
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["success"] is True
    assert data["total_processed"] == 2
    assert data["successful_predictions"] == 2
    assert len(data["results"]) == 2
