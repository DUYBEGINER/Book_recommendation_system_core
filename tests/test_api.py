import pytest
from fastapi.testclient import TestClient
from serve import app
from src.models.hybrid import HybridRecommender
from src.api import routes
import pandas as pd
from pathlib import Path

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_model():
    # Create minimal mock model for testing
    model = HybridRecommender(alpha=0.6)
    routes.recommender = model
    return model

def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"

def test_recommendations_endpoint(client, mock_model):
    response = client.get("/api/v1/recommendations?user_id=1&limit=5")
    assert response.status_code in [200, 503]  # 503 if model not loaded

def test_similar_endpoint(client, mock_model):
    response = client.get("/api/v1/similar?book_id=1&limit=5")
    assert response.status_code in [200, 503]

def test_feedback_endpoint(client):
    payload = {
        "user_id": 1,
        "book_id": 1,
        "event": "view"
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "recorded"

def test_invalid_limit(client):
    response = client.get("/api/v1/recommendations?user_id=1&limit=200")
    assert response.status_code == 400

def test_invalid_event(client):
    payload = {
        "user_id": 1,
        "book_id": 1,
        "event": "invalid"
    }
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 422  # Validation error