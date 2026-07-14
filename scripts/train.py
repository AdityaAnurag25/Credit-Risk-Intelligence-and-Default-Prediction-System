import argparse
import logging
from datetime import datetime, timezone

import joblib
from sklearn.metrics import brier_score_loss

from credit_risk.config import settings
from credit_risk.data import clean_and_label, drop_leaky_columns, load_raw_data
from credit_risk.features import build_features
from credit_risk.models import (
    evaluate,
    fit_isotonic_calibration,
    plot_calibration_curve,
    prepare_model_matrix,
    train,
)
from credit_risk.validation import per_vintage_auc, time_based_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_CHOICES = tuple(sorted(("logistic", "random_forest", "xgboost")))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a credit risk default model.")
    parser.add_argument(
        "--model", choices=MODEL_CHOICES, required=True, help="Which model to train."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(settings.raw_data_path),
        help="Path to raw loan.csv.",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.7, help="Fraction of vintages used for training."
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.15, help="Fraction of vintages held out for validation."
    )
    return parser.parse_args(argv)


def run(model_name: str, data_path: str, train_frac: float, val_frac: float) -> dict:
    logger.info("Loading raw data from %s", data_path)
    df = load_raw_data(data_path)

    logger.info("Cleaning and labeling %d rows", len(df))
    df = clean_and_label(df)

    logger.info("Dropping leaky columns")
    df = drop_leaky_columns(df)

    logger.info("Building features on %d rows", len(df))
    df = build_features(df)

    logger.info(
        "Splitting by issue_d vintage: train=%.0f%%, val=%.0f%%, test=%.0f%%",
        train_frac * 100,
        val_frac * 100,
        (1 - train_frac - val_frac) * 100,
    )
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

    logger.info("Preparing model matrix")
    X, y = prepare_model_matrix(df, target_column=settings.target_column)

    X_train, y_train = X.loc[split.train_index], y.loc[split.train_index]
    X_val, y_val = X.loc[split.val_index], y.loc[split.val_index]
    X_test, y_test = X.loc[split.test_index], y.loc[split.test_index]

    logger.info("Training %s on %d rows", model_name, len(X_train))
    model = train(X_train, y_train, model_name)

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
    logger.info("Brier score — raw: %.4f | isotonic-calibrated: %.4f", brier_raw, brier_calibrated)
    metrics["brier_score_raw"] = brier_raw
    metrics["brier_score_calibrated"] = brier_calibrated

    figure_path = settings.outputs_dir / "figures" / f"calibration_{model_name}.png"
    plot_calibration_curve(
        y_test,
        {"uncalibrated": y_prob, "isotonic": y_prob_calibrated},
        output_path=figure_path,
    )
    logger.info("Saved calibration curve to %s", figure_path)

    models_dir = settings.outputs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = models_dir / f"{model_name}_{timestamp}.pkl"
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        model_name=args.model,
        data_path=args.data_path,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
