from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str | Path,
    target_column: str = "target_default",
) -> Path:
    """Generate an HTML data/target drift report comparing `current_df` against `reference_df`.

    Uses Evidently's `DataDriftPreset` twice: once over the feature columns, and once
    scoped to just `target_column` — the direct replacement for Evidently's
    `TargetDriftPreset`, which was removed in the library's 0.7.x API rewrite (target
    drift is now expressed as `DataDriftPreset` scoped to that one column rather than
    a separate preset class).

    Args:
        reference_df: Baseline distribution to compare against (e.g. training data).
        current_df: Distribution being checked for drift (e.g. recent production data).
        output_path: File path the HTML report is written to. Parent dirs are created.
        target_column: Name of the target column, checked for drift separately from
            the other features if present in both frames.

    Returns:
        `output_path`, for convenience chaining.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Evidently can't compute drift for a column with no data at all in either frame
    # (some raw export columns, e.g. LendingClub's `id`, are entirely empty) — drop
    # those rather than letting the report crash on them.
    empty_columns = {
        col
        for col in reference_df.columns
        if reference_df[col].isna().all()
        or (col in current_df.columns and current_df[col].isna().all())
    }
    if empty_columns:
        reference_df = reference_df.drop(
            columns=[c for c in empty_columns if c in reference_df.columns]
        )
        current_df = current_df.drop(columns=[c for c in empty_columns if c in current_df.columns])

    feature_columns = [c for c in reference_df.columns if c != target_column]
    metrics = [DataDriftPreset(columns=feature_columns)]

    if target_column in reference_df.columns and target_column in current_df.columns:
        metrics.append(DataDriftPreset(columns=[target_column]))

    report = Report(metrics=metrics)
    snapshot = report.run(current_data=current_df, reference_data=reference_df)
    snapshot.save_html(str(output_path))

    return output_path
