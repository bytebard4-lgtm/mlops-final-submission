"""
MLOps Training Pipeline (FINAL STABLE VERSION)
--------------------------------------------
- CI/CD safe
- GitHub Actions safe
- Kubernetes safe
- No MLflow permission issues
- Manual artifact logging
"""

import os
import joblib
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# SAFE ENV FOR CI/CD + K8S
# ==========================================
os.environ["HOME"] = "/tmp"

os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"


# ==========================================
# DATA LOADING
# ==========================================
def load_data():

    df = pd.read_csv(
        "data/iris_custom.csv"
    )

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ==========================================
# TRAINING PIPELINE
# ==========================================
def train():

    # LOCAL MLFLOW DIRECTORY
    base_dir = os.path.abspath(
        os.getcwd()
    )

    mlruns_dir = os.path.join(
        base_dir,
        "mlruns"
    )

    os.makedirs(
        mlruns_dir,
        exist_ok=True
    )

    # FORCE LOCAL TRACKING
    mlflow.set_tracking_uri(
        f"file:{mlruns_dir}"
    )

    # CREATE EXPERIMENT
    experiment_name = (
        "iris-classification"
    )

    try:

        mlflow.create_experiment(
            experiment_name
        )

    except Exception:

        pass

    mlflow.set_experiment(
        experiment_name
    )

    # LOAD DATA
    X_train, X_test, y_train, y_test = (
        load_data()
    )

    # MODELS
    models = {

        "LogisticRegression":
            LogisticRegression(
                max_iter=200
            ),

        "DecisionTree":
            DecisionTreeClassifier(),

        "RandomForest":
            RandomForestClassifier()
    }

    best_model = None

    best_accuracy = -1

    best_name = None

    # ==========================================
    # TRAIN MODELS
    # ==========================================
    for name, model in models.items():

        with mlflow.start_run(
            run_name=name
        ):

            # TRAIN
            model.fit(
                X_train,
                y_train
            )

            # PREDICT
            preds = model.predict(
                X_test
            )

            # METRICS
            acc = accuracy_score(
                y_test,
                preds
            )

            print(
                f"{name} accuracy: "
                f"{acc:.4f}"
            )

            # ==========================================
            # LOG PARAMETERS
            # ==========================================
            mlflow.log_param(
                "model_name",
                name
            )

            mlflow.log_param(
                "dataset",
                "iris_custom.csv"
            )

            # ==========================================
            # LOG METRICS
            # ==========================================
            mlflow.log_metric(
                "accuracy",
                acc
            )

            mlflow.log_metric(
                "error_rate",
                1 - acc
            )

            # ==========================================
            # SAVE MODEL TEMPORARILY
            # ==========================================
            os.makedirs(
                "temp_models",
                exist_ok=True
            )

            temp_model_path = (
                f"temp_models/{name}.pkl"
            )

            joblib.dump(
                model,
                temp_model_path
            )

            # ==========================================
            # LOG MODEL FILE
            # ==========================================
            mlflow.log_artifact(
                temp_model_path
            )

            # ==========================================
            # TRACK BEST MODEL
            # ==========================================
            if acc > best_accuracy:

                best_accuracy = acc

                best_model = model

                best_name = name

    # ==========================================
    # SAVE BEST MODEL
    # ==========================================
    os.makedirs(
        "models",
        exist_ok=True
    )

    best_model_path = (
        "models/best_model.pkl"
    )

    joblib.dump(
        best_model,
        best_model_path
    )

    # ==========================================
    # FINAL RESULTS
    # ==========================================
    print("\n======================")

    print(
        f"BEST MODEL: {best_name}"
    )

    print(
        f"ACCURACY: "
        f"{best_accuracy:.4f}"
    )

    print(
        f"SAVED: "
        f"{best_model_path}"
    )

    print("======================\n")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    train()
