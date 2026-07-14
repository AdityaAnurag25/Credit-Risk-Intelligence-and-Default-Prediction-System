import pandas as pd


def add_income_to_loan_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["income_to_loan_ratio"] = df["annual_inc"] / df["loan_amnt"]
    return df


def add_revol_util_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revol_util_bucket"] = pd.cut(
        df["revol_util"],
        bins=[0, 30, 60, 90, 150],
        labels=["low", "moderate", "high", "very_high"],
    )
    return df


def add_dti_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dti_bucket"] = pd.cut(
        df["dti"],
        bins=[0, 10, 20, 30, 100],
        labels=["low_dti", "moderate_dti", "high_dti", "very_high_dti"],
    )
    return df


def add_employment_stability(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["emp_length_clean"] = df["emp_length"].astype(str).str.extract(r"(\d+)")
    df["emp_length_clean"] = pd.to_numeric(df["emp_length_clean"], errors="coerce")
    df["emp_length_clean"] = df["emp_length_clean"].fillna(0)

    df["employment_stability"] = pd.cut(
        df["emp_length_clean"],
        bins=[-1, 1, 5, 10, 50],
        labels=["new", "short_term", "medium_term", "long_term"],
    )
    return df


def add_credit_history_years(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")

    df["credit_history_years"] = (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365
    return df


def add_inquiry_risk_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["inquiry_risk_bucket"] = pd.cut(
        df["inq_last_6mths"],
        bins=[-1, 1, 3, 6, 100],
        labels=["low_inquiry", "moderate_inquiry", "high_inquiry", "very_high_inquiry"],
    )
    return df


def add_delinquency_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delinquency_risk"] = pd.cut(
        df["delinq_2yrs"],
        bins=[-1, 0, 2, 5, 100],
        labels=["no_delinquency", "low_delinquency", "moderate_delinquency", "high_delinquency"],
    )
    return df


def add_loan_size_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_size_bucket"] = pd.qcut(
        df["loan_amnt"],
        q=4,
        labels=["small", "medium", "large", "very_large"],
    )
    return df


def add_revol_bal_to_income(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["revol_bal_to_income"] = df["revol_bal"] / df["annual_inc"]
    return df


def add_installment_to_income(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["installment_to_income"] = (df["installment"] * 12) / df["annual_inc"]
    return df


def add_credit_exposure_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["credit_exposure_score"] = (
        df["revol_util"].fillna(0) + df["dti"].fillna(0) + df["inq_last_6mths"].fillna(0)
    )
    return df


FEATURE_STEPS = (
    add_income_to_loan_ratio,
    add_revol_util_bucket,
    add_dti_bucket,
    add_employment_stability,
    add_credit_history_years,
    add_inquiry_risk_bucket,
    add_delinquency_risk,
    add_loan_size_bucket,
    add_revol_bal_to_income,
    add_installment_to_income,
    add_credit_exposure_score,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    for step in FEATURE_STEPS:
        df = step(df)
    return df
