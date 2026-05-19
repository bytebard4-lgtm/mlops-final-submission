"""
MLOps Training Pipeline (FINAL VERSION)
---------------------------------------
- Loads dataset from CSV (data ingestion)
- Trains multiple ML models
- Tracks experiments using MLflow (local file store)
- Selects best model based on accuracy
- Saves best model for Flask deployment
- Logs metrics and parameters for comparison
- Supports Continuous Training (CT)
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# =========================
# DATA LOADING (INGESTION)
# =========================
def load_data():
    """
    Load dataset from CSV file.
    Simulates real-world data ingestion stage.
    """

    df = pd.read_csv("data/iris_custom.csv")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# =========================
# TRAINING PIPELINE
# =========================
def train():
    """
    Train multiple models and track experiments using MLflow.
    Selects best model and exports it for deployment.
    """

    # Local MLflow tracking (file-based)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("iris-classification-experiment")

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    best_model = None
    best_accuracy = 0
    best_name = ""

    # =========================
    # TRAIN ALL MODELS
    # =========================
    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            # Train
            model.fit(X_train, y_train)

            # Predict
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            print(f"{name} accuracy: {acc:.4f}")

            # =========================
            # MLflow tracking
            # =========================
            mlflow.log_param("model_name", name)
            mlflow.log_param("dataset", "iris_custom.csv")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("error_rate", 1 - acc)

            mlflow.sklearn.log_model(model, "model")

            # Track best model
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_name = name

    # =========================
    # SAVE BEST MODEL FOR FLASK
    # =========================
    os.makedirs("models", exist_ok=True)

    model_path = "models/best_model.pkl"
    joblib.dump(best_model, model_path)

    # =========================
    # FINAL OUTPUT
    # =========================
    print("\n======================")
    print(f"BEST MODEL: {best_name}")
    print(f"ACCURACY: {best_accuracy:.4f}")
    print(f"MODEL SAVED: {model_path}")
    print("======================\n")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    train()
