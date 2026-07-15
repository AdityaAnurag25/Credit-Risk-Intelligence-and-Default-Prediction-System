import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true: pd.Series, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: the maximum separation between the score
    distributions of the two classes.

        KS = max_t |F1(t) - F0(t)|

    where F1(t) and F0(t) are the cumulative distribution functions of the score
    for the positive (default) and negative (non-default) classes respectively.

    Computed here via the standard identity KS = max(TPR - FPR) over the ROC
    curve's thresholds, which is equivalent to the definition above.

    Args:
        y_true: Binary target values (1 = default).
        y_score: Predicted probability (or score) of the positive class.

    Returns:
        The KS statistic, in [0, 1]. Higher means better separation.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true: pd.Series, y_score: np.ndarray) -> float:
    """Gini coefficient, the standard credit-scoring rescaling of ROC-AUC.

        Gini = 2 * AUC - 1

    Args:
        y_true: Binary target values (1 = default).
        y_score: Predicted probability (or score) of the positive class.

    Returns:
        The Gini coefficient, in [-1, 1]. Higher means better rank-ordering.
    """
    return 2 * roc_auc_score(y_true, y_score) - 1


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between an expected (baseline) and actual distribution.

        PSI = sum_i (actual_pct_i - expected_pct_i) * ln(actual_pct_i / expected_pct_i)

    Bin edges are the quantiles of `expected`, so each expected bin holds ~1/bins of
    the baseline population; `actual` is binned against those same edges. Empty bins
    are floored to a small epsilon so the log term stays finite.

    Rule of thumb: PSI < 0.1 is stable, 0.1-0.25 is a moderate shift worth
    investigating, > 0.25 is a significant shift.

    Args:
        expected: Baseline distribution (e.g. train-set scores or feature values).
        actual: Comparison distribution (e.g. test-set scores or feature values).
        bins: Number of quantile bins to compute the index over.

    Returns:
        The PSI value. 0.0 for identical distributions; grows with the shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    epsilon = 1e-6
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def decile_lift_table(y_true: pd.Series, y_score: np.ndarray) -> pd.DataFrame:
    """Decile lift table: ranks rows into score deciles (decile 1 = highest score /
    highest predicted risk) and reports the event rate and lift of each.

        lift_i = event_rate_i / overall_event_rate

    A well-ranked model concentrates events in the low-numbered (high-score) deciles,
    with lift well above 1 there and cumulative capture approaching 100% quickly.

    Args:
        y_true: Binary target values (1 = default).
        y_score: Predicted probability (or score) of the positive class.

    Returns:
        A DataFrame with one row per decile (1-10) and columns `n`, `n_events`,
        `event_rate`, `lift`, `cumulative_events`, `cumulative_pct_events`, and
        `cumulative_lift`.
    """
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_score": np.asarray(y_score)})
    frame = frame.sort_values("y_score", ascending=False).reset_index(drop=True)
    frame["decile"] = pd.qcut(frame.index, 10, labels=False) + 1

    overall_rate = frame["y_true"].mean()

    table = (
        frame.groupby("decile").agg(n=("y_true", "size"), n_events=("y_true", "sum")).reset_index()
    )
    table["event_rate"] = table["n_events"] / table["n"]
    table["lift"] = table["event_rate"] / overall_rate
    table["cumulative_events"] = table["n_events"].cumsum()
    table["cumulative_pct_events"] = table["cumulative_events"] / table["n_events"].sum()
    table["cumulative_n"] = table["n"].cumsum()
    table["cumulative_lift"] = (table["cumulative_events"] / table["cumulative_n"]) / overall_rate

    return table
