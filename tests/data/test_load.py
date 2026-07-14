import pandas as pd

from credit_risk.data.load import load_raw_data


def test_load_raw_data_reads_csv(tmp_path, raw_loans_df: pd.DataFrame):
    csv_path = tmp_path / "loan.csv"
    raw_loans_df.to_csv(csv_path, index=False)

    loaded = load_raw_data(csv_path)

    assert loaded.shape == raw_loans_df.shape
    assert list(loaded.columns) == list(raw_loans_df.columns)


def test_load_raw_data_accepts_str_path(tmp_path, raw_loans_df: pd.DataFrame):
    csv_path = tmp_path / "loan.csv"
    raw_loans_df.to_csv(csv_path, index=False)

    loaded = load_raw_data(str(csv_path))

    assert len(loaded) == len(raw_loans_df)
