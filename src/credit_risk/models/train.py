import contextlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from credit_risk.config import settings

# Mirrors notebooks/05_model_training.ipynb cell dc2cadd3 exactly. This is
# today's leakage exposure, not a corrected one — a full leakage audit is separate follow-up work.
DROP_COLUMNS: tuple[str, ...] = (
    # Target leakage / post-loan servicing
    "loan_status",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    # Recovery / collection leakage
    "recoveries",
    "collection_recovery_fee",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "total_pymnt",
    "total_pymnt_inv",
    "out_prncp",
    "out_prncp_inv",
    # Settlement leakage
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    # Identifiers / metadata
    "id",
    "member_id",
    "url",
    # High-cardinality noisy text
    "emp_title",
    # Raw date columns
    "issue_d",
    "earliest_cr_line",
)


def prepare_model_matrix(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)

    # build_features() produces genuine pandas `category` dtype columns
    # (pd.cut/pd.qcut buckets). The original notebook pipeline flattened
    # these to plain strings implicitly, via a CSV round-trip between the
    # feature-engineering and training notebooks. This pipeline has no such
    # round-trip, so flatten explicitly to keep the rest of this function's
    # dtype handling identical to the notebook's.
    category_cols = df.select_dtypes(include="category").columns
    for col in category_cols:
        df[col] = df[col].astype(str)

    for col in df.columns:
        if col == target_column:
            continue
        with contextlib.suppress(ValueError, TypeError):
            df[col] = pd.to_numeric(df[col])

    for col in df.columns:
        if col == target_column:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown")

    feature_cols = [c for c in df.columns if c != target_column]
    cat_cols = df[feature_cols].select_dtypes(include=["object", "str"]).columns
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y


def _build_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def _build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=settings.random_seed,
        class_weight="balanced",
    )


def _build_xgboost(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=settings.random_seed,
        scale_pos_weight=scale_pos_weight,
    )


MODEL_BUILDERS = {
    "logistic": _build_logistic,
    "random_forest": _build_random_forest,
    "xgboost": _build_xgboost,
}


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Compute XGBoost's `scale_pos_weight` as the negative-to-positive class ratio.

    Returns 1.0 (no reweighting) if there are no positive examples, since the
    ratio is undefined and XGBoost requires a finite value.
    """
    class_counts = y_train.value_counts()
    negative = class_counts.get(0, 0)
    positive = class_counts.get(1, 0)
    if positive == 0:
        return 1.0
    return negative / positive


def train(X_train: pd.DataFrame, y_train: pd.Series, model_name: str):
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Choose from: {sorted(MODEL_BUILDERS)}"
        )

    if model_name == "xgboost":
        model = _build_xgboost(scale_pos_weight=compute_scale_pos_weight(y_train))
    else:
        model = MODEL_BUILDERS[model_name]()

    model.fit(X_train, y_train)
    return model
