import mlflow.sklearn

from credit_risk.config import settings


def load_champion():
    """Load the model registered under the "champion" alias in the MLflow model registry.

    This is what the serving API loads at startup.

    Returns:
        The deployed model (an sklearn-compatible estimator exposing `predict_proba`).
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.mlflow_registered_model_name}@champion"
    return mlflow.sklearn.load_model(model_uri)
