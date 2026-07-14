import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

RISK_BAND_BINS: list[float] = [0, 0.2, 0.4, 0.6, 0.8, 1]
RISK_BAND_LABELS: list[str] = [
    "Very Low Risk",
    "Low Risk",
    "Medium Risk",
    "High Risk",
    "Very High Risk",
]

DEFAULT_THRESHOLDS = np.arange(0.1, 1.0, 0.05)


def assign_risk_band(probability_default: pd.Series) -> pd.Series:
    return pd.cut(probability_default, bins=RISK_BAND_BINS, labels=RISK_BAND_LABELS)


def sweep_thresholds(y_true: pd.Series, y_prob: pd.Series, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    results = []
    for threshold in thresholds:
        preds = np.where(y_prob >= threshold, 1, 0)
        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
                "f1_score": f1_score(y_true, preds, zero_division=0),
            }
        )
    return pd.DataFrame(results)


def assign_decision(
    probability_default: pd.Series,
    approve_below: float = 0.20,
    reject_above: float = 0.50,
) -> pd.Series:
    conditions = [probability_default < approve_below, probability_default >= reject_above]
    choices = ["Approve", "Reject"]
    return pd.Series(
        np.select(conditions, choices, default="Manual Review"),
        index=probability_default.index,
        name="decision",
    )


def build_scorecard(y_true: pd.Series, y_prob: pd.Series) -> pd.DataFrame:
    scorecard = pd.DataFrame({"probability_default": y_prob, "actual": y_true})
    scorecard["risk_score"] = ((1 - scorecard["probability_default"]) * 1000).astype(int)
    scorecard["risk_band"] = assign_risk_band(scorecard["probability_default"])
    scorecard["decision"] = assign_decision(scorecard["probability_default"])
    return scorecard[["risk_score", "probability_default", "risk_band", "decision", "actual"]]
