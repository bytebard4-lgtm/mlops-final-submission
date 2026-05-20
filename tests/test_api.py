"""
Unit tests for the Flask prediction API.
Tests the /health and /predict endpoints without a live server
by using Flask's built-in test client.
"""

import sys
import os
import joblib
import numpy as np
import pytest

# Make sure the app module can find models/best_model.pkl
os.environ.setdefault("PYTHONPATH", ".")


# ── Fixture: create a minimal model so the app can import ────────────────────
@pytest.fixture(scope="module", autouse=True)
def ensure_model(tmp_path_factory):
    """Train a tiny model so app.py can load it during tests."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris

    os.makedirs("models", exist_ok=True)
    if not os.path.exists("models/best_model.pkl"):
        iris = load_iris()
        clf = LogisticRegression(max_iter=200)
        clf.fit(iris.data, iris.target)
        joblib.dump(clf, "models/best_model.pkl")


# ── Import app after model is guaranteed to exist ────────────────────────────
@pytest.fixture(scope="module")
def client():
    # Import here so the fixture above runs first
    sys.path.insert(0, os.path.abspath("."))
    from app.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_endpoint(client):
    """GET /health must return 200 and model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_endpoint(client):
    """POST /predict with valid Iris features must return a prediction."""
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post(
        "/predict",
        json=payload,
        content_type="application/json"
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)
    assert len(data["prediction"]) == 1


def test_predict_returns_valid_class(client):
    """Prediction must be one of the three Iris classes (0, 1, 2)."""
    payload = {"features": [6.3, 3.3, 6.0, 2.5]}
    response = client.post("/predict", json=payload)
    data = response.get_json()
    assert data["prediction"][0] in [0, 1, 2]
