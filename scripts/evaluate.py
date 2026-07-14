import argparse
import logging

import joblib
from sklearn.model_selection import train_test_split

from credit_risk.config import settings
from credit_risk.data import clean_and_label, drop_leaky_columns, load_raw_data
from credit_risk.features import build_features
from credit_risk.models import evaluate, prepare_model_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved credit risk model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a saved .pkl model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(settings.raw_data_path),
        help="Path to raw loan.csv.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args(argv)


def run(model_path: str, data_path: str, test_size: float) -> dict:
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Rebuilding the holdout split from %s", data_path)
    df = load_raw_data(data_path)
    df = clean_and_label(df)
    df = drop_leaky_columns(df)
    df = build_features(df)

    # Reproduces the holdout split rather than persisting one, so this only lines up with the
    # model's actual training split if --test-size and settings.random_seed match what was used
    # for training. There's no split ID stored alongside the saved model to verify this against.
    X, y = prepare_model_matrix(df, target_column=settings.target_column)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=settings.random_seed,
        stratify=y,
    )

    logger.info("Evaluating on %d holdout rows", len(X_test))
    metrics = evaluate(model, X_test, y_test)
    logger.info("Metrics: %s", metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(model_path=args.model_path, data_path=args.data_path, test_size=args.test_size)


if __name__ == "__main__":
    main()
