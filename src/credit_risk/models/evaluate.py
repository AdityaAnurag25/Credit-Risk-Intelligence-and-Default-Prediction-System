import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from credit_risk.validation.metrics import decile_lift_table, gini_coefficient, ks_statistic


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return {
        "roc_auc": roc_auc_score(y, y_prob),
        "ks_statistic": ks_statistic(y, y_prob),
        "gini_coefficient": gini_coefficient(y, y_prob),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "decile_lift_table": decile_lift_table(y, y_prob).to_dict(orient="records"),
    }
