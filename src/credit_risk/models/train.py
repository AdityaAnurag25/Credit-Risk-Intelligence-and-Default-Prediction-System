import contextlib
import json
import logging
from datetime import UTC, datetime

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from credit_risk.config import settings
from credit_risk.data import clean_and_label, drop_leaky_columns, load_raw_data
from credit_risk.features import build_features
from credit_risk.models.calibrate import fit_isotonic_calibration, plot_calibration_curve
from credit_risk.models.evaluate import evaluate
from credit_risk.models.explain import global_importance, save_shap_plots, top_feature_names
from credit_risk.validation import per_vintage_auc, psi, time_based_split

logger = logging.getLogger(__name__)

TOP_N_FEATURES_FOR_PSI = 5
SHAP_SAMPLE_SIZE = 1000

# Raw categorical columns exposed on the API's LoanApplication schema (the
# application-time subset of the model's features — see credit_risk.api.schemas).
# Their possible values come from the dataset, not a fixed code list, so — unlike
# the pd.cut/pd.qcut bucket columns in credit_risk.features.engineering — a
# vocabulary snapshot has to be persisted at training time for the API to encode
# them consistently with what LabelEncoder saw during training.
API_CATEGORICAL_FIELDS: tuple[str, ...] = (
    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type",
)

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
    cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns
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


def _model_config(model) -> dict[str, str]:
    """Flatten a fitted model's hyperparameters into an MLflow-loggable string dict."""
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    return {f"model_param_{key}": str(value) for key, value in estimator.get_params().items()}


def _category_vocabulary(df: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, list[str]]:
    """Snapshot the sorted-unique vocabulary LabelEncoder would fit for each column.

    `prepare_model_matrix` fits a fresh `LabelEncoder` on every call rather than
    persisting one, so there's nothing the serving API can load to reproduce a
    category's training-time integer code. This reproduces `LabelEncoder`'s own
    `classes_ = sorted(unique(values))` logic (after the same NaN -> "unknown" fill
    `prepare_model_matrix` applies) so the API can look up the same code by position.
    """
    return {col: sorted(df[col].fillna("unknown").astype(str).unique().tolist()) for col in columns}


def run_training(
    model_name: str,
    data_path: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> dict:
    """Run the full training pipeline for one model, tracked as one MLflow run.

    Loads and cleans the raw data, drops leakage columns, builds features, splits
    chronologically by `issue_d`, trains `model_name`, evaluates it on the test
    vintage, calibrates it on the validation vintage, and computes SHAP global
    importance. Logs params (model config, split boundaries, feature count), metrics
    (AUC, KS, Gini, Brier, PSI), and artifacts (calibration curve, SHAP plots, feature
    importance CSV) to MLflow. If `model_name` is "xgboost", the calibrated model is
    also registered in the MLflow model registry as `settings.mlflow_registered_model_name`
    under the "champion" alias.

    Args:
        model_name: One of "logistic", "random_forest", "xgboost".
        data_path: Path to the raw loan.csv.
        train_frac: Fraction of vintages used for training.
        val_frac: Fraction of vintages held out for validation (calibration).

    Returns:
        The metrics dict (same shape as `evaluate()`, plus `per_vintage_auc`,
        `brier_score_raw`, `brier_score_calibrated`, `score_psi`, and `feature_psi`).
    """
    logger.info("Loading raw data from %s", data_path)
    df = load_raw_data(data_path)

    logger.info("Cleaning and labeling %d rows", len(df))
    df = clean_and_label(df)

    logger.info("Dropping leaky columns")
    df = drop_leaky_columns(df)

    logger.info("Building features on %d rows", len(df))
    df = build_features(df)

    vocab = _category_vocabulary(df, API_CATEGORICAL_FIELDS)
    settings.category_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    settings.category_vocab_path.write_text(json.dumps(vocab, indent=2))
    logger.info("Saved category vocabulary to %s", settings.category_vocab_path)

    split = time_based_split(
        df, date_column=settings.issue_date_column, train_frac=train_frac, val_frac=val_frac
    )
    logger.info(
        "Vintage boundaries — train: %s to %s | val: %s to %s | test: %s to %s",
        split.train_start.date(),
        split.train_end.date(),
        split.val_start.date(),
        split.val_end.date(),
        split.test_start.date(),
        split.test_end.date(),
    )

    X, y = prepare_model_matrix(df, target_column=settings.target_column)
    X_train, y_train = X.loc[split.train_index], y.loc[split.train_index]
    X_val, y_val = X.loc[split.val_index], y.loc[split.val_index]
    X_test, y_test = X.loc[split.test_index], y.loc[split.test_index]

    # Per-column training median, in prepare_model_matrix's already-encoded space (so
    # this doubles as the correct default for LabelEncoded categorical columns too, not
    # just numeric ones — reusing its encoding rather than reimplementing LabelEncoder's
    # class-to-code mapping). This is what the serving API fills in for the ~90 model
    # columns a loan application doesn't realistically cover (bureau-report detail,
    # joint-applicant fields, free text) — see credit_risk.api.features.
    feature_defaults = X_train.median().to_dict()
    settings.feature_defaults_path.parent.mkdir(parents=True, exist_ok=True)
    settings.feature_defaults_path.write_text(json.dumps(feature_defaults, indent=2))
    logger.info("Saved feature defaults to %s", settings.feature_defaults_path)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    with mlflow.start_run(run_name=f"{model_name}_{timestamp}"):
        logger.info("Training %s on %d rows", model_name, len(X_train))
        model = train(X_train, y_train, model_name)

        mlflow.log_params(
            {
                "model_name": model_name,
                "n_features": X.shape[1],
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "train_frac": train_frac,
                "val_frac": val_frac,
                "train_start": str(split.train_start.date()),
                "train_end": str(split.train_end.date()),
                "val_start": str(split.val_start.date()),
                "val_end": str(split.val_end.date()),
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
                **_model_config(model),
            }
        )
        mlflow.log_artifact(str(settings.category_vocab_path))
        mlflow.log_artifact(str(settings.feature_defaults_path))

        logger.info("Evaluating on %d holdout rows", len(X_test))
        metrics = evaluate(model, X_test, y_test)
        logger.info("Metrics: %s", metrics)

        test_dates = df.loc[split.test_index, settings.issue_date_column]
        y_prob = model.predict_proba(X_test)[:, 1]
        vintage_auc = per_vintage_auc(test_dates, y_test, y_prob)
        logger.info("Per-vintage AUC (test set):\n%s", vintage_auc.to_string())
        metrics["per_vintage_auc"] = vintage_auc.to_dict()

        logger.info("Calibrating on %d validation rows", len(X_val))
        calibrated_model = fit_isotonic_calibration(model, X_val, y_val)
        y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

        brier_raw = brier_score_loss(y_test, y_prob)
        brier_calibrated = brier_score_loss(y_test, y_prob_calibrated)
        logger.info(
            "Brier score — raw: %.4f | isotonic-calibrated: %.4f", brier_raw, brier_calibrated
        )
        metrics["brier_score_raw"] = brier_raw
        metrics["brier_score_calibrated"] = brier_calibrated

        train_prob = model.predict_proba(X_train)[:, 1]
        score_psi = psi(train_prob, y_prob)
        logger.info("Score PSI (train vs test): %.4f", score_psi)
        metrics["score_psi"] = score_psi

        top_features = top_feature_names(model, list(X_train.columns), TOP_N_FEATURES_FOR_PSI)
        feature_psi = {
            feature: psi(
                X_train[feature].to_numpy(dtype=float), X_test[feature].to_numpy(dtype=float)
            )
            for feature in top_features
        }
        logger.info(
            "Feature PSI (train vs test, top %d features): %s", len(top_features), feature_psi
        )
        metrics["feature_psi"] = feature_psi

        mlflow.log_metrics(
            {
                "roc_auc": metrics["roc_auc"],
                "ks_statistic": metrics["ks_statistic"],
                "gini_coefficient": metrics["gini_coefficient"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "brier_score_raw": brier_raw,
                "brier_score_calibrated": brier_calibrated,
                "score_psi": score_psi,
                **{f"feature_psi_{feature}": value for feature, value in feature_psi.items()},
            }
        )

        figures_dir = settings.outputs_dir / "figures"
        calibration_path = plot_calibration_curve(
            y_test,
            {"uncalibrated": y_prob, "isotonic": y_prob_calibrated},
            output_path=figures_dir / f"calibration_{model_name}.png",
        )
        logger.info("Saved calibration curve to %s", calibration_path)
        mlflow.log_artifact(str(calibration_path))

        X_shap_sample = X_train.sample(
            n=min(SHAP_SAMPLE_SIZE, len(X_train)), random_state=settings.random_seed
        )
        importance = global_importance(model, X_shap_sample)
        importance_path = (
            settings.outputs_dir / "predictions" / f"feature_importance_{model_name}.csv"
        )
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance.to_csv(importance_path, index=False)
        logger.info("Saved feature importance to %s", importance_path)
        mlflow.log_artifact(str(importance_path))

        summary_path, bar_path = save_shap_plots(model, X_shap_sample, figures_dir)
        logger.info("Saved SHAP plots to %s and %s", summary_path, bar_path)
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(bar_path))

        models_dir = settings.outputs_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        logger.info("Saved model to %s", model_path)

        model_info = mlflow.sklearn.log_model(
            calibrated_model, name="model", serialization_format="pickle"
        )

        if model_name == "xgboost":
            registered = mlflow.register_model(
                model_info.model_uri, settings.mlflow_registered_model_name
            )
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(
                settings.mlflow_registered_model_name, "champion", registered.version
            )
            logger.info(
                "Registered %s v%s as 'champion'",
                settings.mlflow_registered_model_name,
                registered.version,
            )

    return metrics
