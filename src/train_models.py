"""
MLOps Training Pipeline (FINAL FIXED VERSION)
--------------------------------------------
- CI/CD safe (GitHub Actions compatible)
- MLflow safe paths (/tmp + repo-local mlruns)
- Fixes /home permission error
- Data ingestion
- Model training
- Experiment tracking
- Best model selection
- Model promotion (pickle)
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 🔥 SAFE ENV CONFIG (MUST BE AFTER IMPORTS)
os.environ["HOME"] = "/tmp"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
os.environ["MLFLOW_ARTIFACT_ROOT"] = "./mlruns"


# =========================
# DATA LOADING
# =========================
def load_data():
    df = pd.read_csv("data/iris_custom.csv")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# TRAINING PIPELINE
# =========================
def train():

    # 🔥 SAFE MLflow directory inside repo
    BASE_DIR = os.path.abspath(os.getcwd())
    mlflow_dir = os.path.join(BASE_DIR, "mlruns")
    os.makedirs(mlflow_dir, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment("iris-classification-experiment")

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    best_model = None
    best_accuracy = -1
    best_name = None


    # =========================
    # TRAIN MODELS
    # =========================
    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            print(f"{name} accuracy: {acc:.4f}")

            # MLflow logging
            mlflow.log_param("model_name", name)
            mlflow.log_param("dataset", "iris_custom.csv")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("error_rate", 1 - acc)

            # 🔥 FIX: prevents /home artifact fallback
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=X_train.iloc[:1],
                artifact_path="model"
            )

            # Track best model
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_name = name


    # =========================
    # SAVE BEST MODEL
    # =========================
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pkl"

    joblib.dump(best_model, model_path)

    print("\n======================")
    print(f"BEST MODEL: {best_name}")
    print(f"ACCURACY: {best_accuracy:.4f}")
    print(f"SAVED: {model_path}")
    print("======================\n")


if __name__ == "__main__":
    train()
