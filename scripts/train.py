import argparse
import logging

from credit_risk.config import settings
from credit_risk.models import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_training(
        model_name=args.model,
        data_path=args.data_path,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
