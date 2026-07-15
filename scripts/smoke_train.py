import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

from credit_risk.config import settings
from credit_risk.data import clean_and_label, drop_leaky_columns, load_raw_data
from credit_risk.features import build_features
from credit_risk.models import prepare_model_matrix, train
from credit_risk.validation import time_based_split

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_lending.csv"
AUC_THRESHOLD = 0.6


def run(data_path: Path = FIXTURE_PATH, auc_threshold: float = AUC_THRESHOLD) -> float:
    """Run the core training pipeline on a small dataset and return test-set AUC.

    A CI sanity gate, not a real accuracy benchmark: it exists to catch pipeline
    breakage (a column rename, a step that now crashes on edge-case input, ...),
    not to validate modelling quality. Skips MLflow logging, calibration, and SHAP
    — those are exercised by scripts/train.py's own tests, not this smoke check.
    """
    df = load_raw_data(data_path)
    df = clean_and_label(df)
    df = drop_leaky_columns(df)
    df = build_features(df)

    split = time_based_split(df, date_column=settings.issue_date_column)
    X, y = prepare_model_matrix(df, target_column=settings.target_column)
    X_train, y_train = X.loc[split.train_index], y.loc[split.train_index]
    X_test, y_test = X.loc[split.test_index], y.loc[split.test_index]

    model = train(X_train, y_train, "xgboost")
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"Smoke training AUC on {data_path.name}: {auc:.4f}")
    if auc <= auc_threshold:
        print(
            f"FAILED: AUC {auc:.4f} did not exceed the sanity threshold ({auc_threshold})",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"OK: AUC {auc:.4f} exceeds the sanity threshold ({auc_threshold})")
    return auc


if __name__ == "__main__":
    run()
