import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from credit_risk.models.explain import global_importance, local_explanation, save_shap_plots


def _tiny_fitted_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + X["b"] > 1).astype(int)
    model = XGBClassifier(n_estimators=5, max_depth=2).fit(X, y)
    return model, X


def test_global_importance_shape_and_features_non_empty():
    model, X = _tiny_fitted_model()

    importance = global_importance(model, X)

    assert list(importance.columns) == ["feature", "mean_abs_shap"]
    assert len(importance) == X.shape[1]
    assert set(importance["feature"]) == set(X.columns)
    assert not importance.empty
    assert (importance["mean_abs_shap"] >= 0).all()
    # Sorted descending.
    assert importance["mean_abs_shap"].is_monotonic_decreasing


def test_local_explanation_shape_and_features_non_empty():
    model, X = _tiny_fitted_model()

    result = local_explanation(model, X.iloc[[0]], n=3)

    assert list(result.columns) == ["feature", "value", "shap_value", "direction", "magnitude"]
    assert len(result) == 3
    assert not result.empty
    assert set(result["feature"]) <= set(X.columns)
    assert set(result["direction"]) <= {"increases_risk", "decreases_risk"}
    assert result["magnitude"].is_monotonic_decreasing


def test_local_explanation_rejects_multi_row_input():
    model, X = _tiny_fitted_model()

    try:
        local_explanation(model, X.iloc[:2])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for multi-row x_row")


def test_save_shap_plots_creates_both_pngs(tmp_path):
    model, X = _tiny_fitted_model()

    summary_path, bar_path = save_shap_plots(model, X, tmp_path)

    assert summary_path.exists()
    assert bar_path.exists()
    assert summary_path.stat().st_size > 0
    assert bar_path.stat().st_size > 0
