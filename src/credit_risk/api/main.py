import logging
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI

from credit_risk.api.decision import assign_decision
from credit_risk.api.features import build_feature_row, load_category_vocab, load_feature_defaults
from credit_risk.api.risk_bands import assign_risk_band
from credit_risk.api.schemas import LoanApplication, ScoreResponse, TopReason
from credit_risk.config import settings
from credit_risk.models.explain import local_explanation
from credit_risk.models.load import load_champion

logger = logging.getLogger(__name__)

TOP_REASONS_COUNT = 5

# Populated by the lifespan handler at startup; read by request handlers. FastAPI
# runs each request in its own task against the same app instance, so this is safe
# as a plain dict as long as nothing mutates it after startup.
model_state: dict[str, Any] = {}


def _unwrap_tree_estimator(calibrated_model):
    """Get back the raw XGBoost estimator wrapped inside the champion model.

    `run_training()` only ever registers a `CalibratedClassifierCV(FrozenEstimator(xgb))`
    as champion, so this structure is a safe assumption here. Explaining the raw
    tree model directly (rather than the calibrated wrapper) gets the fast, exact
    `TreeExplainer` path instead of a slow model-agnostic one — see
    `credit_risk.models.explain._explain`. Isotonic calibration is a monotonic
    transform of the base model's score, so the raw model's feature attributions
    are a valid explanation of the calibrated PD's ranking.
    """
    return calibrated_model.calibrated_classifiers_[0].estimator.estimator


def _resolve_model_version() -> str:
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(
        settings.mlflow_registered_model_name, "champion"
    )
    return str(model_version.version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading champion model from the MLflow registry")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model = load_champion()
    model_state["model"] = model
    model_state["explainer_model"] = _unwrap_tree_estimator(model)
    model_state["feature_names"] = list(model.feature_names_in_)
    model_state["category_vocab"] = load_category_vocab()
    model_state["feature_defaults"] = load_feature_defaults()
    model_state["model_version"] = _resolve_model_version()
    logger.info("Champion model loaded (version %s)", model_state["model_version"])

    yield

    model_state.clear()


app = FastAPI(title="Credit Risk PD Scoring API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": model_state.get("model_version", "unknown")}


@app.post("/score", response_model=ScoreResponse)
def score(application: LoanApplication) -> ScoreResponse:
    row = build_feature_row(
        application,
        model_state["feature_names"],
        model_state["category_vocab"],
        model_state["feature_defaults"],
    )

    probability_default = float(model_state["model"].predict_proba(row)[0, 1])

    reasons = local_explanation(model_state["explainer_model"], row, n=TOP_REASONS_COUNT)
    submitted = application.model_dump()
    top_reasons = [
        TopReason(
            feature=reason.feature,
            # Show the value the applicant actually submitted where available (e.g.
            # sub_grade="B3"), falling back to the model's encoded numeric value for
            # the ~90 columns the request doesn't cover.
            value=submitted.get(reason.feature, reason.value),
            direction=reason.direction,
            magnitude=reason.magnitude,
        )
        for reason in reasons.itertuples()
    ]

    return ScoreResponse(
        pd=probability_default,
        risk_band=assign_risk_band(probability_default),
        decision=assign_decision(probability_default),
        top_reasons=top_reasons,
    )
