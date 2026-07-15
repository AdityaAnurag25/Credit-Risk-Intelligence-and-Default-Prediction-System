import numpy as np
import pytest

from credit_risk.validation.metrics import (
    decile_lift_table,
    gini_coefficient,
    ks_statistic,
    psi,
)


def test_ks_statistic_perfect_separation_is_one():
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    assert ks_statistic(y_true, y_score) == pytest.approx(1.0)


def test_ks_statistic_matches_manual_cdf_calculation():
    # Negatives: 0.1, 0.2, 0.3, 0.4 | Positives: 0.3, 0.4, 0.5, 0.6
    # max|F0(t) - F1(t)| is 0.5, reached at t=0.2 (F0=0.5, F1=0) and t=0.4 (F0=1.0, F1=0.5).
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.6]

    assert ks_statistic(y_true, y_score) == pytest.approx(0.5)


def test_ks_statistic_no_separation_is_near_zero():
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_score = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    assert ks_statistic(y_true, y_score) == pytest.approx(0.0)


def test_gini_coefficient_perfect_separation_is_one():
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    assert gini_coefficient(y_true, y_score) == pytest.approx(1.0)


def test_gini_coefficient_matches_known_auc():
    # Positives: 0.35, 0.8 | Negatives: 0.1, 0.4
    # AUC = fraction of pos/neg pairs ranked correctly = 3/4 = 0.75 -> Gini = 2*0.75 - 1 = 0.5
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]

    assert gini_coefficient(y_true, y_score) == pytest.approx(0.5)


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)

    assert psi(values, values, bins=10) == pytest.approx(0.0, abs=1e-9)


def test_psi_matches_manual_calculation():
    # bins=2 on `expected` puts the breakpoint at the median (1.5), splitting the five 1s
    # from the five 2s. `actual` shifts mass from bin 2 into bin 1 (9 vs 1 instead of 5 vs 5).
    expected = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    actual = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])

    result = psi(expected, actual, bins=2)

    assert result == pytest.approx(0.87889, abs=1e-4)


def test_psi_is_positive_for_shifted_distributions():
    rng = np.random.default_rng(0)
    expected = rng.normal(loc=0, scale=1, size=500)
    actual = rng.normal(loc=2, scale=1, size=500)

    assert psi(expected, actual, bins=10) > 0.25


def test_decile_lift_table_has_expected_shape_and_columns():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_score = rng.random(size=100)

    table = decile_lift_table(y_true, y_score)

    assert len(table) == 10
    assert list(table["decile"]) == list(range(1, 11))
    assert {
        "n",
        "n_events",
        "event_rate",
        "lift",
        "cumulative_events",
        "cumulative_pct_events",
        "cumulative_lift",
    } <= set(table.columns)


def test_decile_lift_table_matches_known_values():
    # 10 rows, one per decile by construction. The 3 events are the 3 highest scores,
    # so decile 1-3 each fully capture one event and deciles 4-10 capture none.
    y_score = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    table = decile_lift_table(y_true, y_score)

    assert list(table["n_events"]) == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert table.loc[table["decile"] == 1, "event_rate"].iloc[0] == pytest.approx(1.0)
    assert table.loc[table["decile"] == 1, "lift"].iloc[0] == pytest.approx(1 / 0.3)
    assert table.loc[table["decile"] == 3, "cumulative_pct_events"].iloc[0] == pytest.approx(1.0)
    assert table.loc[table["decile"] == 10, "cumulative_pct_events"].iloc[0] == pytest.approx(1.0)
