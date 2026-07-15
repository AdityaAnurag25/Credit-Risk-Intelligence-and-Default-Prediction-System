import argparse
import logging

import joblib

from credit_risk.config import settings
from credit_risk.data import clean_and_label, drop_leaky_columns, load_raw_data
from credit_risk.features import build_features
from credit_risk.models import (
    global_importance,
    local_explanation,
    prepare_model_matrix,
    save_shap_plots,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SHAP_SAMPLE_SIZE = 1000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain a saved credit risk model with SHAP.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to a saved .pkl model.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(settings.raw_data_path),
        help="Path to raw loan.csv.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SHAP_SAMPLE_SIZE,
        help="Number of rows to sample for SHAP (never run SHAP on the full dataset).",
    )
    return parser.parse_args(argv)


def run(model_path: str, data_path: str, sample_size: int) -> dict:
    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Rebuilding features from %s", data_path)
    df = load_raw_data(data_path)
    df = clean_and_label(df)
    df = drop_leaky_columns(df)
    df = build_features(df)

    X, _ = prepare_model_matrix(df, target_column=settings.target_column)
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=settings.random_seed)
    logger.info("Sampled %d of %d rows for SHAP", len(X_sample), len(X))

    importance = global_importance(model, X_sample)
    logger.info("Global SHAP importance (top 15):\n%s", importance.head(15).to_string())

    figures_dir = settings.outputs_dir / "figures"
    summary_path, bar_path = save_shap_plots(model, X_sample, figures_dir)
    logger.info("Saved SHAP summary plot to %s", summary_path)
    logger.info("Saved SHAP bar plot to %s", bar_path)

    example_row = X_sample.iloc[[0]]
    local = local_explanation(model, example_row)
    logger.info("Local explanation for one sampled row:\n%s", local.to_string())

    return {"global_importance": importance, "local_explanation": local}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(model_path=args.model_path, data_path=args.data_path, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
