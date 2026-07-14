import pandas as pd

from credit_risk.data import clean_and_label
from credit_risk.data.leakage import LEAKY_COLUMNS, drop_leaky_columns


def _fixture_with_all_leaky_columns() -> pd.DataFrame:
    data = {
        "loan_status": ["Fully Paid", "Charged Off", "Fully Paid", "Charged Off"],
        "loan_amnt": [1000, 2000, 1500, 3000],
        "grade": ["A", "B", "C", "D"],
    }
    for col in LEAKY_COLUMNS:
        data[col] = [1, 2, 3, 4]
    return pd.DataFrame(data)


def test_drop_leaky_columns_removes_all_known_leaky_columns():
    df = _fixture_with_all_leaky_columns()
    result = drop_leaky_columns(df)

    survivors = set(LEAKY_COLUMNS) & set(result.columns)
    assert not survivors, f"Leaky columns survived: {survivors}"


def test_drop_leaky_columns_preserves_safe_columns():
    df = _fixture_with_all_leaky_columns()
    result = drop_leaky_columns(df)

    assert {"loan_status", "loan_amnt", "grade"} <= set(result.columns)


def test_drop_leaky_columns_ignores_missing_columns():
    df = pd.DataFrame({"loan_amnt": [1, 2], "grade": ["A", "B"]})
    result = drop_leaky_columns(df)

    assert list(result.columns) == ["loan_amnt", "grade"]


def test_drop_leaky_columns_logs_dropped_columns(caplog):
    df = _fixture_with_all_leaky_columns()
    with caplog.at_level("INFO"):
        drop_leaky_columns(df)

    assert "recoveries" in caplog.text
    assert "hardship_status" in caplog.text


def test_no_leaky_column_survives_the_cleaning_pipeline():
    df = _fixture_with_all_leaky_columns()
    cleaned = clean_and_label(df)
    result = drop_leaky_columns(cleaned)

    survivors = set(LEAKY_COLUMNS) & set(result.columns)
    assert not survivors, f"Leaky columns survived the cleaning pipeline: {survivors}"
