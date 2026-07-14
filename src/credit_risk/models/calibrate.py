from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator


def fit_isotonic_calibration(
    model, X_val: pd.DataFrame, y_val: pd.Series
) -> CalibratedClassifierCV:
    """Calibrate a fitted classifier's probabilities with isotonic regression on a validation set.

    Wraps `model` in `FrozenEstimator` so `CalibratedClassifierCV` treats it as
    already-fitted and calibrates directly on `(X_val, y_val)` instead of refitting
    or cross-validating the base model. This is the modern replacement for the
    removed `cv="prefit"` option: `scikit-learn>=1.6` raises on `cv="prefit"` and
    the version pinned in this project (`>=1.8`) has dropped it entirely.

    Args:
        model: An already-fitted classifier exposing `predict_proba`.
        X_val: Validation feature matrix. Must be disjoint from the data `model`
            was trained on — the caller is responsible for that separation.
        y_val: Validation target values.

    Returns:
        A fitted `CalibratedClassifierCV` wrapping `model`, calibrated on `(X_val, y_val)`.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(X_val, y_val)
    return calibrated


def plot_calibration_curve(
    y_true: pd.Series,
    probability_sets: dict[str, np.ndarray],
    output_path: Path,
    n_bins: int = 10,
) -> Path:
    """Plot a reliability diagram for one or more sets of predicted probabilities.

    Args:
        y_true: Binary target values.
        probability_sets: Mapping of curve label (e.g. "uncalibrated", "isotonic")
            to predicted probability of the positive class.
        output_path: File path the figure is saved to. Parent directories are created.
        n_bins: Number of bins used to compute each reliability curve.

    Returns:
        `output_path`, for convenience chaining.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    for label, y_prob in probability_sets.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        ax.plot(mean_predicted_value, fraction_of_positives, marker="o", label=label)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path
