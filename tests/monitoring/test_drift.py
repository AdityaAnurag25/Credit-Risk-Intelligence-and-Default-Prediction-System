import numpy as np
import pandas as pd

from credit_risk.monitoring.drift import generate_drift_report


def _sample_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "loan_amnt": rng.uniform(1000, 30000, size=n),
            "grade": rng.choice(["A", "B", "C", "D"], size=n),
            "target_default": rng.integers(0, 2, size=n),
        }
    )


def test_generate_drift_report_creates_html(tmp_path):
    reference_df = _sample_frame(200, seed=0)
    current_df = _sample_frame(200, seed=1)
    output_path = tmp_path / "drift.html"

    result = generate_drift_report(reference_df, current_df, output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_drift_report_drops_entirely_empty_columns(tmp_path):
    # Some raw export columns (e.g. LendingClub's `id`) are entirely NaN in practice;
    # Evidently can't compute drift for those and used to crash on them.
    reference_df = _sample_frame(200, seed=0)
    current_df = _sample_frame(200, seed=1)
    reference_df["empty_col"] = np.nan
    current_df["empty_col"] = np.nan
    output_path = tmp_path / "drift.html"

    result = generate_drift_report(reference_df, current_df, output_path)

    assert result.exists()


def test_generate_drift_report_without_target_column(tmp_path):
    reference_df = _sample_frame(200, seed=0).drop(columns=["target_default"])
    current_df = _sample_frame(200, seed=1).drop(columns=["target_default"])
    output_path = tmp_path / "drift.html"

    result = generate_drift_report(reference_df, current_df, output_path)

    assert result.exists()


def test_generate_drift_report_creates_parent_directories(tmp_path):
    reference_df = _sample_frame(50, seed=0)
    current_df = _sample_frame(50, seed=1)
    output_path = tmp_path / "nested" / "dir" / "drift.html"

    result = generate_drift_report(reference_df, current_df, output_path)

    assert result.exists()
