import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from credit_risk.api.schemas import LoanApplication
from credit_risk.config import settings
from credit_risk.features import build_features
from credit_risk.features.engineering import (
    DELINQUENCY_RISK_LABELS,
    DTI_BUCKET_LABELS,
    EMPLOYMENT_STABILITY_LABELS,
    INQUIRY_RISK_BUCKET_LABELS,
    LOAN_SIZE_BUCKET_LABELS,
    REVOL_UTIL_BUCKET_LABELS,
)

# add_loan_size_bucket() uses pd.qcut on loan_amnt, which needs enough spread in the
# data to compute quartiles — a lone scoring request can't provide that on its own.
# Padding with reference amounts spanning LendingClub's typical range lets qcut run;
# only the first (real) row is kept afterward. This means loan_size_bucket is placed
# against this small reference set rather than the true training-set quartiles — a
# minor approximation for a feature that isn't among the model's top SHAP drivers.
_LOAN_AMNT_PADDING: tuple[float, ...] = (1000.0, 8000.0, 15000.0, 25000.0, 40000.0)

# Fixed-vocabulary bucket columns build_features() derives — their category labels
# are hardcoded in credit_risk.features.engineering and never change independently
# of that code, so (unlike the raw categorical inputs below) no persisted training
# snapshot is needed to encode them consistently.
_FIXED_BUCKET_VOCAB: dict[str, list[str]] = {
    "revol_util_bucket": sorted(REVOL_UTIL_BUCKET_LABELS),
    "dti_bucket": sorted(DTI_BUCKET_LABELS),
    "employment_stability": sorted(EMPLOYMENT_STABILITY_LABELS),
    "inquiry_risk_bucket": sorted(INQUIRY_RISK_BUCKET_LABELS),
    "delinquency_risk": sorted(DELINQUENCY_RISK_LABELS),
    "loan_size_bucket": sorted(LOAN_SIZE_BUCKET_LABELS),
}


def load_category_vocab(path: Path | None = None) -> dict[str, list[str]]:
    """Load the raw-categorical-column vocabulary snapshot written by `run_training`."""
    path = path or settings.category_vocab_path
    return json.loads(path.read_text())


def load_feature_defaults(path: Path | None = None) -> dict[str, float]:
    """Load the per-column training-median defaults snapshot written by `run_training`."""
    path = path or settings.feature_defaults_path
    return json.loads(path.read_text())


def build_feature_row(
    application: LoanApplication,
    feature_names: list[str],
    category_vocab: dict[str, list[str]],
    feature_defaults: dict[str, float],
) -> pd.DataFrame:
    """Convert one `LoanApplication` into a single-row feature matrix ready for
    `model.predict_proba`, aligned to the model's exact training-time columns.

    Only the application-time ("safe") fields on `LoanApplication` are real inputs.
    `issue_d` is set to today — scoring happens at decision time, not some historical
    issue date — so `credit_history_years` reflects a live application. Every other
    model column the request doesn't cover (bureau-report detail, joint-applicant
    fields, free text — see `LoanApplication`'s docstring) is filled with its
    per-column training median from `feature_defaults`. This matters for categorical
    columns specifically: filling with a bare 0 would encode as whichever category
    happens to sort first alphabetically under `LabelEncoder`, not as "typical" or
    "missing" — the training median (in `prepare_model_matrix`'s already-encoded
    space) is a meaningfully safer default.

    Args:
        application: The incoming loan application.
        feature_names: The model's expected input columns, in order
            (`model.feature_names_in_`).
        category_vocab: Sorted-unique vocabulary for the raw categorical inputs,
            from `load_category_vocab()` — used to reproduce the integer code
            `LabelEncoder` assigned each category at training time.
        feature_defaults: Per-column training median, from `load_feature_defaults()`.

    Returns:
        A one-row DataFrame with columns `feature_names`, fully numeric.
    """
    payload = application.model_dump()
    payload["issue_d"] = datetime.now(UTC).strftime("%b-%Y")

    padded = pd.concat(
        [pd.DataFrame([payload]), pd.DataFrame({"loan_amnt": _LOAN_AMNT_PADDING})],
        ignore_index=True,
    )
    engineered = build_features(padded)

    category_cols = engineered.select_dtypes(include="category").columns
    for col in category_cols:
        engineered[col] = engineered[col].astype(str)

    row = engineered.iloc[[0]].reindex(columns=feature_names)

    vocab = {**category_vocab, **_FIXED_BUCKET_VOCAB}
    for column, classes in vocab.items():
        if column not in row.columns:
            continue
        raw_value = row.at[row.index[0], column]
        value = str(raw_value) if pd.notna(raw_value) else "unknown"
        row[column] = classes.index(value) if value in classes else -1

    row = row.apply(pd.to_numeric, errors="coerce")
    return row.fillna(value=feature_defaults).fillna(0.0)
