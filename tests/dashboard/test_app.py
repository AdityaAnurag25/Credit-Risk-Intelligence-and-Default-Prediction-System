from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from credit_risk.config import settings
from credit_risk.models.train import run_training

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_lending.csv"
APP_PATH = Path(__file__).resolve().parents[2] / "streamlit_app.py"


@pytest.fixture
def champion_env(tmp_path, monkeypatch):
    """Train a real (tiny) champion into an isolated models_dir/mlflow store.

    Exercises the exact code path `make train` uses in production — including
    `export_champion_artifact` and `export_serving_encoders` — rather than
    hand-crafting joblib/json fixtures that could drift from what training
    actually produces.
    """
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")
    monkeypatch.setattr(settings, "outputs_dir", tmp_path / "outputs")
    monkeypatch.setattr(
        settings, "category_vocab_path", tmp_path / "outputs" / "models" / "category_vocab.json"
    )
    monkeypatch.setattr(
        settings,
        "feature_defaults_path",
        tmp_path / "outputs" / "models" / "feature_defaults.json",
    )
    monkeypatch.setattr(settings, "mlflow_tracking_uri", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(settings, "mlflow_experiment_name", "test_dashboard_experiment")
    monkeypatch.setattr(settings, "mlflow_registered_model_name", "test_dashboard_model")

    run_training(model_name="xgboost", data_path=str(FIXTURE_PATH), train_frac=0.7, val_frac=0.15)


def test_dashboard_shows_error_when_no_champion_model(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "models_dir", tmp_path / "models")

    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()

    assert not at.exception
    assert any("No champion model found" in error.value for error in at.error)


def test_dashboard_scores_an_application_end_to_end(champion_env):
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()

    assert not at.exception
    assert not at.error

    at.button[0].click().run()

    assert not at.exception
    metric_labels = [metric.label for metric in at.metric]
    assert "Probability of default" in metric_labels
    assert "Risk band" in metric_labels
    assert "Decision" in metric_labels
