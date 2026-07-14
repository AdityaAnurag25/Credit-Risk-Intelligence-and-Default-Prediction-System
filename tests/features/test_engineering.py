import numpy as np
import pandas as pd

from credit_risk.features.engineering import (
    add_credit_exposure_score,
    add_credit_history_years,
    add_delinquency_risk,
    add_employment_stability,
    add_income_to_loan_ratio,
    add_inquiry_risk_bucket,
    add_installment_to_income,
    add_loan_size_bucket,
    add_revol_bal_to_income,
    add_revol_util_bucket,
    add_dti_bucket,
)

ZERO_INCOME_ROW = 1
MISSING_VALUES_ROW = 2
EXTREME_DTI_ROW = 3


def test_income_to_loan_ratio_normal(raw_features_df: pd.DataFrame):
    out = add_income_to_loan_ratio(raw_features_df)
    assert out["income_to_loan_ratio"].iloc[0] == 60000 / 10000


def test_income_to_loan_ratio_zero_income_is_zero_not_error(raw_features_df: pd.DataFrame):
    out = add_income_to_loan_ratio(raw_features_df)
    assert out["income_to_loan_ratio"].iloc[ZERO_INCOME_ROW] == 0


def test_income_to_loan_ratio_missing_income_propagates_nan(raw_features_df: pd.DataFrame):
    out = add_income_to_loan_ratio(raw_features_df)
    assert np.isnan(out["income_to_loan_ratio"].iloc[MISSING_VALUES_ROW])


def test_revol_bal_to_income_zero_income_is_inf(raw_features_df: pd.DataFrame):
    out = add_revol_bal_to_income(raw_features_df)
    assert np.isinf(out["revol_bal_to_income"].iloc[ZERO_INCOME_ROW])


def test_installment_to_income_zero_income_is_inf(raw_features_df: pd.DataFrame):
    out = add_installment_to_income(raw_features_df)
    assert np.isinf(out["installment_to_income"].iloc[ZERO_INCOME_ROW])


def test_revol_util_bucket_assigns_expected_labels(raw_features_df: pd.DataFrame):
    out = add_revol_util_bucket(raw_features_df)
    assert out["revol_util_bucket"].iloc[0] == "moderate"  # revol_util=45 -> (30,60]


def test_revol_util_bucket_missing_value_is_nan(raw_features_df: pd.DataFrame):
    out = add_revol_util_bucket(raw_features_df)
    assert pd.isna(out["revol_util_bucket"].iloc[MISSING_VALUES_ROW])


def test_dti_bucket_normal_value(raw_features_df: pd.DataFrame):
    out = add_dti_bucket(raw_features_df)
    assert out["dti_bucket"].iloc[0] == "moderate_dti"  # dti=18 -> (10,20]


def test_dti_bucket_extreme_value_falls_outside_bins(raw_features_df: pd.DataFrame):
    # bins cap at 100; a dti of 999 has nowhere to land and becomes NaN
    out = add_dti_bucket(raw_features_df)
    assert pd.isna(out["dti_bucket"].iloc[EXTREME_DTI_ROW])


def test_dti_bucket_missing_value_is_nan(raw_features_df: pd.DataFrame):
    out = add_dti_bucket(raw_features_df)
    assert pd.isna(out["dti_bucket"].iloc[MISSING_VALUES_ROW])


def test_employment_stability_parses_numeric_prefix(raw_features_df: pd.DataFrame):
    out = add_employment_stability(raw_features_df)
    assert out["emp_length_clean"].iloc[3] == 10  # "10+ years"
    assert out["emp_length_clean"].iloc[4] == 1  # "< 1 year"
    assert out["employment_stability"].iloc[3] == "medium_term"
    assert out["employment_stability"].iloc[4] == "new"


def test_employment_stability_missing_emp_length_defaults_to_zero(raw_features_df: pd.DataFrame):
    out = add_employment_stability(raw_features_df)
    assert out["emp_length_clean"].iloc[MISSING_VALUES_ROW] == 0
    assert out["employment_stability"].iloc[MISSING_VALUES_ROW] == "new"


def test_credit_history_years_normal(raw_features_df: pd.DataFrame):
    out = add_credit_history_years(raw_features_df)
    # issue_d Jan-2019, earliest_cr_line Jan-2010 -> ~9 years
    assert 8.9 < out["credit_history_years"].iloc[0] < 9.1


def test_credit_history_years_missing_dates_is_nan(raw_features_df: pd.DataFrame):
    out = add_credit_history_years(raw_features_df)
    assert pd.isna(out["credit_history_years"].iloc[MISSING_VALUES_ROW])


def test_inquiry_risk_bucket_labels(raw_features_df: pd.DataFrame):
    out = add_inquiry_risk_bucket(raw_features_df)
    assert out["inquiry_risk_bucket"].iloc[4] == "low_inquiry"  # inq=0
    assert out["inquiry_risk_bucket"].iloc[9] == "moderate_inquiry"  # inq=2
    assert out["inquiry_risk_bucket"].iloc[10] == "very_high_inquiry"  # inq=7


def test_inquiry_risk_bucket_missing_value_is_nan(raw_features_df: pd.DataFrame):
    out = add_inquiry_risk_bucket(raw_features_df)
    assert pd.isna(out["inquiry_risk_bucket"].iloc[MISSING_VALUES_ROW])


def test_delinquency_risk_labels(raw_features_df: pd.DataFrame):
    out = add_delinquency_risk(raw_features_df)
    assert out["delinquency_risk"].iloc[1] == "no_delinquency"  # delinq_2yrs=0
    assert out["delinquency_risk"].iloc[10] == "high_delinquency"  # delinq_2yrs=6


def test_delinquency_risk_missing_value_is_nan(raw_features_df: pd.DataFrame):
    out = add_delinquency_risk(raw_features_df)
    assert pd.isna(out["delinquency_risk"].iloc[MISSING_VALUES_ROW])


def test_loan_size_bucket_assigns_all_rows(raw_features_df: pd.DataFrame):
    out = add_loan_size_bucket(raw_features_df)
    assert out["loan_size_bucket"].notna().all()
    assert set(out["loan_size_bucket"].cat.categories) == {
        "small", "medium", "large", "very_large",
    }


def test_credit_exposure_score_normal(raw_features_df: pd.DataFrame):
    out = add_credit_exposure_score(raw_features_df)
    # revol_util=45 + dti=18 + inq_last_6mths=2
    assert out["credit_exposure_score"].iloc[0] == 45 + 18 + 2


def test_credit_exposure_score_missing_inputs_default_to_zero(raw_features_df: pd.DataFrame):
    # unlike the other transforms, this one fillna(0)s its inputs, so an
    # all-missing row should score 0, not NaN
    out = add_credit_exposure_score(raw_features_df)
    assert out["credit_exposure_score"].iloc[MISSING_VALUES_ROW] == 0
