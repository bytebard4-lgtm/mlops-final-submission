import mlflow
import mlflow.pyfunc

def test_model_load():
    """Check model exists in MLflow runs"""

    mlflow.set_tracking_uri("file:./mlruns")

    client = mlflow.tracking.MlflowClient()

    experiments = client.search_experiments()

    assert len(experiments) > 0, "No experiments found"
