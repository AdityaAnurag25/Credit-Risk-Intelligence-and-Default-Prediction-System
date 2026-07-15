import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_risk.config import settings
from credit_risk.models.load import load_champion


def test_load_champion_falls_back_to_mlflow_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(settings, "mlflow_registered_model_name", "test_credit_risk_model")
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("test_experiment")

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.integers(0, 2, size=50))
    model = LogisticRegression().fit(X, y)

    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(model, name="model", serialization_format="pickle")
        registered = mlflow.register_model(model_info.model_uri, "test_credit_risk_model")
        client = mlflow.MlflowClient()
        client.set_registered_model_alias("test_credit_risk_model", "champion", registered.version)

    loaded = load_champion()

    assert hasattr(loaded, "predict_proba")
    probs = loaded.predict_proba(X)
    assert probs.shape == (50, 2)


def test_load_champion_prefers_local_joblib(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    settings.models_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.integers(0, 2, size=50))
    model = LogisticRegression().fit(X, y)
    joblib.dump(model, settings.models_dir / "champion.joblib")

    loaded = load_champion()

    assert hasattr(loaded, "predict_proba")
    probs = loaded.predict_proba(X)
    assert probs.shape == (50, 2)
