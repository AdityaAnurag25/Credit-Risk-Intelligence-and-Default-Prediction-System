from typing import Literal

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    """Application-time loan details — the model's safe (non-leaky) feature set.

    Only fields available before a loan is issued are collected here, per
    CLAUDE.md's Data Leakage Reference. The underlying model was trained on ~90
    additional columns (bureau-report detail, joint-applicant fields, free text)
    that aren't realistic to collect on a loan application form; those are filled
    with neutral defaults at scoring time — see `credit_risk.api.features`.
    """

    loan_amnt: float = Field(gt=0, description="Requested loan amount, in dollars.")
    term: str = Field(description='Loan term, e.g. "36 months" or "60 months".')
    int_rate: float = Field(ge=0, le=100, description="Interest rate, as a percentage.")
    installment: float = Field(gt=0, description="Monthly payment if the loan is issued.")
    grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(description="LendingClub grade.")
    sub_grade: str = Field(pattern=r"^[A-G][1-5]$", description='Sub-grade, e.g. "B3".')
    emp_length: str | None = Field(
        default=None, description='Employment length, e.g. "5 years", "< 1 year", "10+ years".'
    )
    home_ownership: str = Field(description='e.g. "RENT", "OWN", "MORTGAGE".')
    annual_inc: float = Field(gt=0, description="Self-reported annual income.")
    verification_status: str = Field(
        description='e.g. "Verified", "Source Verified", "Not Verified".'
    )
    purpose: str = Field(description='Loan purpose, e.g. "debt_consolidation".')
    dti: float = Field(ge=0, description="Debt-to-income ratio.")
    delinq_2yrs: int = Field(ge=0, description="Delinquencies in the past 2 years.")
    earliest_cr_line: str = Field(description='Earliest credit line date, e.g. "Jan-2010".')
    inq_last_6mths: int = Field(ge=0, description="Credit inquiries in the last 6 months.")
    mths_since_last_delinq: float | None = Field(default=None, ge=0)
    mths_since_last_record: float | None = Field(default=None, ge=0)
    open_acc: int = Field(ge=0, description="Number of open credit lines.")
    pub_rec: int = Field(ge=0, description="Number of derogatory public records.")
    revol_bal: float = Field(ge=0, description="Total revolving credit balance.")
    revol_util: float = Field(ge=0, le=150, description="Revolving line utilization rate (%).")
    total_acc: int = Field(ge=0, description="Total number of credit lines.")
    initial_list_status: str = Field(description='"w" (whole) or "f" (fractional).')
    application_type: str = Field(description='"Individual" or "Joint App".')


class TopReason(BaseModel):
    """One feature's SHAP contribution to a single prediction, adverse-action style."""

    feature: str
    value: float | str | None
    direction: Literal["increases_risk", "decreases_risk"]
    magnitude: float


class ScoreResponse(BaseModel):
    pd: float = Field(ge=0, le=1, description="Calibrated probability of default.")
    risk_band: str
    decision: str
    top_reasons: list[TopReason]
