import numpy as np
import pandas as pd
import pytest

from credit_risk.validation.splits import per_vintage_auc, time_based_split


def _monthly_df(n_months: int = 10, rows_per_month: int = 10) -> pd.DataFrame:
    months = pd.date_range("2018-01-01", periods=n_months, freq="MS")
    dates = np.repeat(months, rows_per_month)
    rng = np.random.default_rng(0)
    shuffled = pd.Series(dates).sample(frac=1, random_state=0).reset_index(drop=True)
    return pd.DataFrame(
        {
            "issue_d": shuffled.dt.strftime("%b-%Y"),
            "target_default": rng.integers(0, 2, size=len(shuffled)),
        }
    )


def test_split_sizes_match_requested_fractions():
    df = _monthly_df(n_months=10, rows_per_month=10)
    split = time_based_split(df, train_frac=0.7, val_frac=0.15)

    assert len(split.train_index) == 70
    assert len(split.val_index) == 15
    assert len(split.test_index) == 15


def test_split_is_chronological_with_no_overlap():
    df = _monthly_df()
    split = time_based_split(df, train_frac=0.7, val_frac=0.15)

    dates = pd.to_datetime(df["issue_d"], format="%b-%Y")
    train_dates = dates.loc[split.train_index]
    val_dates = dates.loc[split.val_index]
    test_dates = dates.loc[split.test_index]

    assert train_dates.max() <= val_dates.min()
    assert val_dates.max() <= test_dates.min()

    all_index = set(split.train_index) | set(split.val_index) | set(split.test_index)
    assert all_index == set(df.index)
    assert len(all_index) == len(df)


def test_boundaries_match_actual_min_max_per_split():
    df = _monthly_df()
    split = time_based_split(df, train_frac=0.7, val_frac=0.15)

    dates = pd.to_datetime(df["issue_d"], format="%b-%Y")
    assert split.train_start == dates.loc[split.train_index].min()
    assert split.train_end == dates.loc[split.train_index].max()
    assert split.val_start == dates.loc[split.val_index].min()
    assert split.val_end == dates.loc[split.val_index].max()
    assert split.test_start == dates.loc[split.test_index].min()
    assert split.test_end == dates.loc[split.test_index].max()


def test_accepts_already_parsed_datetime_column():
    df = _monthly_df()
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    split = time_based_split(df, train_frac=0.7, val_frac=0.15)

    assert len(split.train_index) + len(split.val_index) + len(split.test_index) == len(df)


@pytest.mark.parametrize(("train_frac", "val_frac"), [(0.7, 0.3), (1.0, 0.1), (0.5, -0.1)])
def test_rejects_invalid_fractions(train_frac, val_frac):
    df = _monthly_df()
    with pytest.raises(ValueError):
        time_based_split(df, train_frac=train_frac, val_frac=val_frac)


def test_rejects_unparseable_dates():
    df = pd.DataFrame({"issue_d": ["not-a-date", "also-not-a-date"], "target_default": [0, 1]})
    with pytest.raises(ValueError):
        time_based_split(df)


def test_per_vintage_auc_computes_auc_per_month():
    dates = pd.to_datetime(["Jan-2019"] * 4 + ["Feb-2019"] * 4, format="%b-%Y")
    y_true = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.9, 0.8, 0.1, 0.2])

    result = per_vintage_auc(dates, y_true, y_prob)

    assert list(result.index) == ["2019-01", "2019-02"]
    assert result["2019-01"] == pytest.approx(1.0)
    assert result["2019-02"] == pytest.approx(0.0)


def test_per_vintage_auc_returns_nan_for_single_class_vintage():
    dates = pd.to_datetime(["Mar-2019"] * 3, format="%b-%Y")
    y_true = pd.Series([0, 0, 0])
    y_prob = np.array([0.1, 0.4, 0.6])

    result = per_vintage_auc(dates, y_true, y_prob)

    assert np.isnan(result["2019-03"])
