"""
Unit tests for the Flask prediction API.
Uses Flask's built-in test client — no live server needed.
"""

import sys
import os
import joblib
import pytest

os.environ.setdefault("PYTHONPATH", ".")


# ── Fixture: ensure a trained model exists before importing app ───────────────
@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    """Train a tiny model so app.py can load it during tests."""
    from sklearn.linear_model import LogisticRegression
    import pandas as pd

    os.makedirs("models", exist_ok=True)
    if not os.path.exists("models/best_model.pkl"):
        df = pd.read_csv("data/iris_custom.csv")
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, y)
        joblib.dump(clf, "models/best_model.pkl")


# ── Flask test client ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, os.path.abspath("."))
    from app.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_health_endpoint(client):
    """GET /health must return 200 and model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_endpoint(client):
    """POST /predict with valid features must return a prediction list."""
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)
    assert len(data["prediction"]) == 1


def test_predict_returns_valid_class(client):
    """Prediction must be one of the three Iris species strings."""
    valid_classes = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
    payload = {"features": [6.3, 3.3, 6.0, 2.5]}
    response = client.post("/predict", json=payload)
    data = response.get_json()
    assert data["prediction"][0] in valid_classes
