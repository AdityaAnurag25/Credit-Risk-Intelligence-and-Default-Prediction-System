from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _parse_dates(series) -> pd.Series:
    series = series if isinstance(series, pd.Series) else pd.Series(series)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, format="%b-%Y", errors="coerce")


@dataclass(frozen=True)
class TimeBasedSplit:
    """Row indices and vintage boundaries for a chronological train/val/test split."""

    train_index: pd.Index
    val_index: pd.Index
    test_index: pd.Index
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def time_based_split(
    df: pd.DataFrame,
    date_column: str = "issue_d",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> TimeBasedSplit:
    """Split a DataFrame chronologically by ``date_column`` into train/val/test.

    Rows are sorted by date, then partitioned by position: the earliest
    ``train_frac`` become train, the next ``val_frac`` become validation, and
    the remainder becomes test. This mirrors how a model would actually be
    deployed — trained on past vintages and scored on future ones.

    Args:
        df: DataFrame containing ``date_column``.
        date_column: Name of the date column to sort and split by. Parsed as
            a datetime (LendingClub's ``%b-%Y`` format) if not already one.
        train_frac: Fraction of rows assigned to train.
        val_frac: Fraction of rows assigned to validation. The remaining
            ``1 - train_frac - val_frac`` becomes test.

    Returns:
        A `TimeBasedSplit` with row indices and vintage boundaries for each split.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must each be in (0, 1) and sum to less than 1")

    dates = _parse_dates(df[date_column])
    if dates.isna().any():
        raise ValueError(f"{date_column!r} contains values that could not be parsed as dates")

    order = dates.sort_values(kind="stable").index
    n = len(order)
    train_end_pos = int(n * train_frac)
    val_end_pos = int(n * (train_frac + val_frac))

    train_index = order[:train_end_pos]
    val_index = order[train_end_pos:val_end_pos]
    test_index = order[val_end_pos:]

    sorted_dates = dates.loc[order]

    return TimeBasedSplit(
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        train_start=sorted_dates.iloc[0],
        train_end=sorted_dates.iloc[train_end_pos - 1],
        val_start=sorted_dates.iloc[train_end_pos],
        val_end=sorted_dates.iloc[val_end_pos - 1],
        test_start=sorted_dates.iloc[val_end_pos],
        test_end=sorted_dates.iloc[-1],
    )


def per_vintage_auc(dates: pd.Series, y_true: pd.Series, y_prob: np.ndarray) -> pd.Series:
    """Compute ROC-AUC separately for each issue year-month vintage.

    Args:
        dates: Issue dates aligned with `y_true`/`y_prob`.
        y_true: Binary target values.
        y_prob: Predicted probability of the positive class.

    Returns:
        A Series indexed by "YYYY-MM" vintage, sorted chronologically. Vintages
        with only one class present are reported as NaN since ROC-AUC is undefined.
    """
    dates = _parse_dates(dates)
    frame = pd.DataFrame(
        {
            "vintage": dates.dt.to_period("M").astype(str).to_numpy(),
            "y_true": np.asarray(y_true),
            "y_prob": np.asarray(y_prob),
        }
    )

    results = {
        vintage: (
            roc_auc_score(group["y_true"], group["y_prob"])
            if group["y_true"].nunique() > 1
            else np.nan
        )
        for vintage, group in frame.groupby("vintage")
    }
    return pd.Series(results).sort_index()
