import argparse
import logging

import joblib
import numpy as np

from credit_risk.config import settings
from credit_risk.data import clean_and_label, drop_leaky_columns, load_raw_data
from credit_risk.features import build_features
from credit_risk.models import evaluate, prepare_model_matrix
from credit_risk.validation import per_vintage_auc, psi, time_based_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOP_N_FEATURES_FOR_PSI = 5


def _top_feature_names(model, feature_names: list[str], n: int) -> list[str]:
    """Pick the `n` features the model relies on most, for covariate-shift PSI.

    Reads `feature_importances_` (tree ensembles) or `coef_` (linear models),
    unwrapping a `Pipeline`'s final step if needed. Falls back to the first `n`
    feature columns, with a warning, if neither is available.
    """
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model

    if hasattr(estimator, "feature_importances_"):
        importances = np.abs(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.ravel(estimator.coef_))
    else:
        logger.warning(
            "Model exposes neither feature_importances_ nor coef_; using the first %d "
            "feature columns for PSI instead of the most important ones.",
            n,
        )
        return list(feature_names[:n])

    top_positions = np.argsort(importances)[::-1][:n]
    return [feature_names[i] for i in top_positions]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved credit risk model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a saved .pkl model.")
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


def run(model_path: str, data_path: str, train_frac: float, val_frac: float) -> dict:
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Rebuilding the time-based holdout split from %s", data_path)
    df = load_raw_data(data_path)
    df = clean_and_label(df)
    df = drop_leaky_columns(df)
    df = build_features(df)

    # Reproduces the split rather than persisting one, so this only lines up with the model's
    # actual training split if --train-frac/--val-frac match what was used for training. There's
    # no split ID stored alongside the saved model to verify this against.
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
    X_train = X.loc[split.train_index]
    X_test, y_test = X.loc[split.test_index], y.loc[split.test_index]

    logger.info("Evaluating on %d holdout rows", len(X_test))
    metrics = evaluate(model, X_test, y_test)
    logger.info("Metrics: %s", metrics)

    test_dates = df.loc[split.test_index, settings.issue_date_column]
    y_prob = model.predict_proba(X_test)[:, 1]
    vintage_auc = per_vintage_auc(test_dates, y_test, y_prob)
    logger.info("Per-vintage AUC (test set):\n%s", vintage_auc.to_string())
    metrics["per_vintage_auc"] = vintage_auc.to_dict()

    train_prob = model.predict_proba(X_train)[:, 1]
    score_psi = psi(train_prob, y_prob)
    logger.info("Score PSI (train vs test): %.4f", score_psi)
    metrics["score_psi"] = score_psi

    top_features = _top_feature_names(model, list(X_train.columns), TOP_N_FEATURES_FOR_PSI)
    feature_psi = {
        feature: psi(X_train[feature].to_numpy(dtype=float), X_test[feature].to_numpy(dtype=float))
        for feature in top_features
    }
    logger.info("Feature PSI (train vs test, top %d features): %s", len(top_features), feature_psi)
    metrics["feature_psi"] = feature_psi

    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        model_path=args.model_path,
        data_path=args.data_path,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
