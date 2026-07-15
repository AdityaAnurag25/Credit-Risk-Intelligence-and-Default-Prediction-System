import logging

import mlflow
import mlflow.sklearn

from credit_risk.config import settings
from credit_risk.models.train import export_champion_artifact, export_serving_encoders

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    version = client.get_model_version_by_alias(settings.mlflow_registered_model_name, "champion")
    run = client.get_run(version.run_id)

    logger.info(
        "Exporting %s v%s (run %s) to a local joblib artifact",
        settings.mlflow_registered_model_name,
        version.version,
        version.run_id,
    )

    model = mlflow.sklearn.load_model(f"models:/{settings.mlflow_registered_model_name}@champion")
    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError(
            "Champion model has no feature_names_in_ (was it fit on a pandas DataFrame?); "
            "cannot export champion_metadata.json without a feature list."
        )

    export_champion_artifact(
        model,
        version=str(version.version),
        feature_names=list(model.feature_names_in_),
        test_auc=run.data.metrics["roc_auc"],
        test_ks=run.data.metrics["ks_statistic"],
        test_gini=run.data.metrics["gini_coefficient"],
        brier=run.data.metrics["brier_score_calibrated"],
    )

    # Assumes outputs/models/{category_vocab,feature_defaults}.json on disk still
    # match this champion's run — true right after training, but stale if other
    # trainings ran since. Same assumption the local/Docker API already makes.
    export_serving_encoders()


if __name__ == "__main__":
    main()
