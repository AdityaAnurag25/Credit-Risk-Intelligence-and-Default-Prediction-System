from credit_risk.config import settings


def assign_decision(probability_default: float) -> str:
    """Map a predicted probability of default to a lending decision.

    Thresholds are read from settings (`decision_approve_below`, `decision_reject_above`,
    tunable via `.env`) rather than hardcoded — the same threshold semantics as
    `credit_risk.models.decision.assign_decision`'s batch version.
    """
    if probability_default < settings.decision_approve_below:
        return "Approve"
    if probability_default >= settings.decision_reject_above:
        return "Reject"
    return "Manual Review"
