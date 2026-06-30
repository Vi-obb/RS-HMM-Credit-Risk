from __future__ import annotations

import pytest
import pandas as pd

from rs_hmm.evaluation import (
    compute_calibration_method_metrics,
    compute_monthly_default_count_error,
    compute_stress_gradient_metrics,
    probability_method_columns,
)


def small_predictions() -> pd.DataFrame:
    base_rows = {
        "split": ["test"] * 6,
        "loan_id": ["a", "b", "c", "d", "e", "f"],
        "month": [1, 1, 1, 2, 2, 2],
        "month_date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01", "2024-02-01"],
        "y": [1, 0, 0, 0, 1, 0],
        "p_stress": [0.55, 0.60, 0.70, 0.88, 0.94, 0.99],
        "hard_stress": [1, 1, 1, 1, 1, 1],
    }
    baseline = pd.DataFrame(
        {
            **base_rows,
            "model": ["baseline_logit"] * 6,
            "p": [0.60, 0.20, 0.20, 0.10, 0.70, 0.10],
            "p_raw": [0.50, 0.30, 0.30, 0.20, 0.60, 0.20],
            "p_isotonic": [0.80, 0.10, 0.10, 0.05, 0.90, 0.05],
        }
    )
    regime = pd.DataFrame(
        {
            **base_rows,
            "model": ["regime_aware_logit"] * 6,
            "p": [0.50, 0.25, 0.25, 0.15, 0.65, 0.15],
            "p_raw": [0.45, 0.35, 0.35, 0.25, 0.55, 0.25],
            "p_isotonic": [0.75, 0.15, 0.15, 0.10, 0.85, 0.10],
        }
    )
    return pd.concat([baseline, regime], ignore_index=True)


def test_stress_gradient_metrics_use_p_stress_when_all_rows_are_hard_stress() -> None:
    predictions = small_predictions()

    metrics = compute_stress_gradient_metrics(predictions, n_bins=3)

    baseline = metrics[metrics["model"] == "baseline_logit"]
    assert len(baseline) == 3
    assert baseline["stress_bin_count"].nunique() == 1
    assert baseline["stress_bin_count"].iloc[0] == 3
    assert set(baseline["hard_stress_share"]) == {1.0}
    assert baseline["event_count"].sum() == 2
    assert baseline["p_stress_min"].min() == pytest.approx(0.55)
    assert baseline["p_stress_max"].max() == pytest.approx(0.99)


def test_monthly_default_count_error_reports_portfolio_timing_error() -> None:
    predictions = small_predictions()

    monthly = compute_monthly_default_count_error(predictions)

    january = monthly[
        (monthly["model"] == "baseline_logit") & (monthly["test_month"] == "2024-01")
    ].iloc[0]
    assert january["n"] == 3
    assert january["observed_defaults"] == pytest.approx(1.0)
    assert january["predicted_defaults"] == pytest.approx(1.0)
    assert january["absolute_count_error"] == pytest.approx(0.0)
    assert january["observed_default_rate"] == pytest.approx(1 / 3)
    assert january["predicted_default_rate"] == pytest.approx(1 / 3)
    assert january["p_stress_mean"] == pytest.approx((0.55 + 0.60 + 0.70) / 3)
    assert january["hard_stress_share"] == pytest.approx(1.0)


def test_calibration_method_metrics_compare_probability_columns_and_ignore_stress() -> None:
    predictions = small_predictions()

    assert probability_method_columns(predictions) == ["p", "p_raw", "p_isotonic"]

    metrics = compute_calibration_method_metrics(predictions, n_bins=3)
    baseline = metrics[metrics["model"] == "baseline_logit"]

    assert set(baseline["calibration_method"]) == {"primary", "raw", "isotonic"}
    assert "p_stress" not in set(metrics["probability_column"])
    assert set(metrics["split"]) == {"test"}
    assert baseline.loc[baseline["calibration_method"] == "isotonic", "brier"].iloc[0] < baseline.loc[
        baseline["calibration_method"] == "raw",
        "brier",
    ].iloc[0]
