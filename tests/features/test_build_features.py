import pandas as pd

from credit_risk.features.engineering import build_features

EXPECTED_NEW_COLUMNS = {
    "income_to_loan_ratio",
    "revol_util_bucket",
    "dti_bucket",
    "emp_length_clean",
    "employment_stability",
    "credit_history_years",
    "inquiry_risk_bucket",
    "delinquency_risk",
    "loan_size_bucket",
    "revol_bal_to_income",
    "installment_to_income",
    "credit_exposure_score",
}


def test_build_features_adds_all_expected_columns(raw_features_df: pd.DataFrame):
    out = build_features(raw_features_df)
    assert EXPECTED_NEW_COLUMNS <= set(out.columns)


def test_build_features_preserves_row_count(raw_features_df: pd.DataFrame):
    out = build_features(raw_features_df)
    assert len(out) == len(raw_features_df)


def test_build_features_does_not_mutate_input(raw_features_df: pd.DataFrame):
    original_columns = list(raw_features_df.columns)
    build_features(raw_features_df)
    assert list(raw_features_df.columns) == original_columns
