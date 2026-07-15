import argparse
import logging

import pandas as pd

from credit_risk.config import settings
from credit_risk.monitoring import generate_drift_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Evidently data/target drift report. Intended to run on a "
        "schedule (see .github/workflows/drift.yml) comparing recent production data "
        "against a fixed reference snapshot."
    )
    parser.add_argument("--reference-path", type=str, default=str(settings.reference_data_path))
    parser.add_argument("--current-path", type=str, default=str(settings.current_data_path))
    parser.add_argument("--output-path", type=str, default=str(settings.drift_report_path))
    return parser.parse_args(argv)


def run(reference_path: str, current_path: str, output_path: str) -> str:
    logger.info("Loading reference data from %s", reference_path)
    reference_df = pd.read_parquet(reference_path)

    logger.info("Loading current data from %s", current_path)
    current_df = pd.read_parquet(current_path)

    logger.info(
        "Generating drift report (%d reference rows, %d current rows)",
        len(reference_df),
        len(current_df),
    )
    report_path = generate_drift_report(
        reference_df, current_df, output_path, target_column=settings.target_column
    )
    logger.info("Saved drift report to %s", report_path)
    return str(report_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        reference_path=args.reference_path,
        current_path=args.current_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
