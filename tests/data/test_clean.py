import pandas as pd

from credit_risk.data.clean import TARGET_STATUSES, clean_and_label


def test_target_mapping_correctness(raw_loans_df: pd.DataFrame):
    cleaned = clean_and_label(raw_loans_df)

    charged_off = cleaned[cleaned["loan_status"] == "Charged Off"]
    fully_paid = cleaned[cleaned["loan_status"] == "Fully Paid"]

    assert (charged_off["target_default"] == 1).all()
    assert (fully_paid["target_default"] == 0).all()
    assert charged_off["target_default"].sum() == len(charged_off)
    assert fully_paid["target_default"].sum() == 0


def test_non_target_statuses_are_dropped(raw_loans_df: pd.DataFrame):
    cleaned = clean_and_label(raw_loans_df)

    dropped_statuses = {"Current", "Late (31-120 days)", "Late (16-30 days)", "In Grace Period", "Default"}

    assert set(cleaned["loan_status"].unique()) == set(TARGET_STATUSES)
    assert dropped_statuses.isdisjoint(cleaned["loan_status"].unique())

    expected_rows = raw_loans_df["loan_status"].isin(TARGET_STATUSES).sum()
    assert len(cleaned) == expected_rows


def test_basic_dtype_handling(raw_loans_df: pd.DataFrame):
    cleaned = clean_and_label(raw_loans_df)

    assert not cleaned["grade"].str.startswith(" ").any()
    assert not cleaned["grade"].str.endswith(" ").any()
    assert set(cleaned["grade"].unique()) <= {"A", "B", "C", "D", "E"}
