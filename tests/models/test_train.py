import numpy as np
import pandas as pd
import pytest

from credit_risk.models.train import compute_scale_pos_weight, train


def test_compute_scale_pos_weight_matches_negative_to_positive_ratio():
    y = pd.Series([0, 0, 0, 0, 1, 1])
    assert compute_scale_pos_weight(y) == pytest.approx(4 / 2)


def test_compute_scale_pos_weight_handles_no_positives():
    y = pd.Series([0, 0, 0])
    assert compute_scale_pos_weight(y) == 1.0


def test_train_xgboost_sets_scale_pos_weight_from_class_ratio():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((100, 3)), columns=["a", "b", "c"])
    y = pd.Series([0] * 90 + [1] * 10)

    model = train(X, y, "xgboost")

    assert model.get_params()["scale_pos_weight"] == pytest.approx(9.0)


def test_train_logistic_uses_balanced_class_weight():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((100, 3)), columns=["a", "b", "c"])
    y = pd.Series([0] * 90 + [1] * 10)

    model = train(X, y, "logistic")

    assert model.named_steps["model"].class_weight == "balanced"


def test_train_unknown_model_name_raises():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    with pytest.raises(ValueError):
        train(X, y, "not_a_model")
