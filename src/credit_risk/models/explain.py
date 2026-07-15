from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _tree_estimator(model):
    """Return the tree-ensemble estimator inside `model`, or None if it isn't one."""
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    return estimator if hasattr(estimator, "feature_importances_") else None


def _explain(model, X: pd.DataFrame, background: pd.DataFrame | None = None) -> shap.Explanation:
    """Build a SHAP Explanation for `model` on `X`, normalized to the positive class.

    Tree ensembles (XGBoost, RandomForest, ...) use the exact, fast `TreeExplainer`
    and need no background sample. Anything else (e.g. a scaler + LogisticRegression
    `Pipeline`) falls back to the generic, model-agnostic `Explainer` driven by
    `predict_proba`, which requires a representative `background` sample to build its
    masker from and is inherently much slower — that's a property of non-tree
    explainers, not something to work around here.
    """
    if _tree_estimator(model) is not None:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    else:
        if background is None or len(background) < 2:
            raise ValueError(
                "This model has no feature_importances_ (not a tree ensemble), so SHAP needs "
                "a representative `background` sample (multiple rows) to build a masker from."
            )
        masker = shap.sample(background, min(100, len(background)))
        explainer = shap.Explainer(model.predict_proba, masker)

    explanation = explainer(X)
    if explanation.values.ndim == 3:
        # (n_samples, n_features, n_classes) -> keep the positive (default) class only.
        explanation = explanation[:, :, -1]

    return explanation


def global_importance(model, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature — a global ranking of how much each
    feature moves the model's predicted probability of default, on average.

        importance_j = mean_i |SHAP(x_i, feature_j)|

    Args:
        model: A fitted classifier (or scaler + classifier `Pipeline`).
        X_sample: Feature matrix to explain. Cap this before calling — SHAP over the
            full training set is not something you want to run — e.g.
            `X.sample(n=1000, random_state=settings.random_seed)`.

    Returns:
        A DataFrame with columns `feature` and `mean_abs_shap`, sorted descending.
    """
    explanation = _explain(model, X_sample, background=X_sample)

    return (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def local_explanation(
    model,
    x_row: pd.DataFrame,
    n: int = 5,
    background: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Top-`n` features driving one prediction, formatted for adverse-action-style output.

    Each row is one feature's SHAP contribution to *this* prediction: `direction` is
    `"increases_risk"` when its SHAP value is positive (pushes the predicted
    probability of default up) and `"decreases_risk"` when negative. Rows are ranked
    by `magnitude` (absolute SHAP value) descending — the order a credit decision
    would cite reasons in.

    Args:
        model: A fitted classifier (or scaler + classifier `Pipeline`).
        x_row: A single-row feature matrix (e.g. `X_sample.iloc[[0]]`).
        n: Number of top features to return.
        background: Representative multi-row sample used to build the SHAP masker.
            Required for non-tree models (see `_explain`); ignored for tree ensembles,
            which don't need one.

    Returns:
        A DataFrame with columns `feature`, `value`, `shap_value`, `direction`, and
        `magnitude`, sorted by `magnitude` descending, with at most `n` rows.
    """
    if len(x_row) != 1:
        raise ValueError(f"x_row must contain exactly one row, got {len(x_row)}")

    explanation = _explain(model, x_row, background=background)
    shap_values = explanation.values[0]

    table = pd.DataFrame(
        {
            "feature": x_row.columns,
            "value": x_row.iloc[0].to_numpy(),
            "shap_value": shap_values,
        }
    )
    table["direction"] = np.where(table["shap_value"] > 0, "increases_risk", "decreases_risk")
    table["magnitude"] = table["shap_value"].abs()

    return table.sort_values("magnitude", ascending=False).head(n).reset_index(drop=True)


def save_shap_plots(model, X_sample: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Save a SHAP beeswarm summary plot and a SHAP bar plot to `output_dir`.

    Args:
        model: A fitted classifier (or scaler + classifier `Pipeline`).
        X_sample: Feature matrix to explain (already capped to a reasonable sample size).
        output_dir: Directory the two PNGs are written to. Created if missing.

    Returns:
        `(summary_plot_path, bar_plot_path)`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explanation = _explain(model, X_sample, background=X_sample)

    summary_path = output_dir / "shap_summary.png"
    plt.figure()
    shap.summary_plot(explanation, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()

    bar_path = output_dir / "shap_bar.png"
    plt.figure()
    shap.summary_plot(explanation, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()

    return summary_path, bar_path
