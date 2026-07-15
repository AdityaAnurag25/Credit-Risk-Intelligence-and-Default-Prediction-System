RISK_BAND_THRESHOLDS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
RISK_BAND_LABELS: tuple[str, ...] = ("Very Low", "Low", "Medium", "High", "Very High")


def assign_risk_band(probability_default: float) -> str:
    """Map a predicted probability of default to a risk band.

    Bands: [0, 0.2) Very Low, [0.2, 0.4) Low, [0.4, 0.6) Medium, [0.6, 0.8) High,
    [0.8, 1] Very High — the same boundaries `credit_risk.models.decision.assign_risk_band`
    uses for batch scoring, for a single value instead of a pandas Series.
    """
    for threshold, label in zip(RISK_BAND_THRESHOLDS, RISK_BAND_LABELS, strict=False):
        if probability_default < threshold:
            return label
    return RISK_BAND_LABELS[-1]
