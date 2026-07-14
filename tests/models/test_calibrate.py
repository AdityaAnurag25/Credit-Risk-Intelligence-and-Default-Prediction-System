import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_risk.models.calibrate import fit_isotonic_calibration, plot_calibration_curve


def _fitted_model_and_val_split():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.random((200, 3)), columns=["a", "b", "c"])
    y_train = pd.Series(rng.integers(0, 2, size=200))
    model = LogisticRegression().fit(X_train, y_train)

    X_val = pd.DataFrame(rng.random((200, 3)), columns=["a", "b", "c"])
    y_val = pd.Series(rng.integers(0, 2, size=200))
    return model, X_val, y_val


def test_fit_isotonic_calibration_returns_valid_probabilities():
    model, X_val, y_val = _fitted_model_and_val_split()

    calibrated = fit_isotonic_calibration(model, X_val, y_val)
    probs = calibrated.predict_proba(X_val)[:, 1]

    assert probs.shape == (len(X_val),)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_fit_isotonic_calibration_does_not_mutate_base_model():
    model, X_val, y_val = _fitted_model_and_val_split()
    original_coef = model.coef_.copy()

    fit_isotonic_calibration(model, X_val, y_val)

    assert np.array_equal(model.coef_, original_coef)


def test_plot_calibration_curve_creates_png_at_output_path(tmp_path):
    y_true = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    probs = {
        "uncalibrated": np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.9]),
    }
    output_path = tmp_path / "figures" / "calibration_test.png"

    result = plot_calibration_curve(y_true, probs, output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_calibration_curve_accepts_multiple_probability_sets(tmp_path):
    y_true = pd.Series([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    probs = {
        "uncalibrated": np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.9]),
        "isotonic": np.array([0.0, 0.1, 1.0, 0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 1.0]),
    }
    output_path = tmp_path / "calibration_multi.png"

    result = plot_calibration_curve(y_true, probs, output_path)

    assert result.exists()
