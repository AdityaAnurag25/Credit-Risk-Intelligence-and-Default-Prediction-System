import argparse
import logging
from datetime import datetime, timezone

import joblib
from sklearn.model_selection import train_test_split

from credit_risk.config import settings
from credit_risk.data import clean_and_label, load_raw_data
from credit_risk.features import build_features
from credit_risk.models import evaluate, prepare_model_matrix, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_CHOICES = tuple(sorted(("logistic", "random_forest", "xgboost")))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a credit risk default model.")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True, help="Which model to train.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(settings.raw_data_path),
        help="Path to raw loan.csv.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args(argv)


def run(model_name: str, data_path: str, test_size: float) -> dict:
    logger.info("Loading raw data from %s", data_path)
    df = load_raw_data(data_path)

    logger.info("Cleaning and labeling %d rows", len(df))
    df = clean_and_label(df)

    logger.info("Building features on %d rows", len(df))
    df = build_features(df)

    logger.info("Preparing model matrix")
    X, y = prepare_model_matrix(df, target_column=settings.target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=settings.random_seed,
        stratify=y,
    )

    logger.info("Training %s on %d rows", model_name, len(X_train))
    model = train(X_train, y_train, model_name)

    logger.info("Evaluating on %d holdout rows", len(X_test))
    metrics = evaluate(model, X_test, y_test)
    logger.info("Metrics: %s", metrics)

    models_dir = settings.outputs_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = models_dir / f"{model_name}_{timestamp}.pkl"
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(model_name=args.model, data_path=args.data_path, test_size=args.test_size)


if __name__ == "__main__":
    main()
