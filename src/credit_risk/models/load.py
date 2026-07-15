import logging

import joblib
import mlflow.sklearn

from credit_risk.config import settings

logger = logging.getLogger(__name__)


def load_champion():
    """Load the deployed champion model.

    Prefers the committed `models/champion.joblib` artifact so the API and
    dashboard can start without an MLflow tracking backend available, e.g. on
    Streamlit Community Cloud. Falls back to the MLflow model registry's
    "champion"-aliased model when no local artifact is present.

    Returns:
        The deployed model (an sklearn-compatible estimator exposing `predict_proba`).
    """
    champion_path = settings.models_dir / "champion.joblib"
    if champion_path.exists():
        logger.info("Loading champion model from %s", champion_path)
        return joblib.load(champion_path)

    logger.info("No local champion.joblib found, loading from MLflow registry")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.mlflow_registered_model_name}@champion"
    return mlflow.sklearn.load_model(model_uri)
