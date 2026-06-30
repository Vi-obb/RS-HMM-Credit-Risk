from __future__ import annotations

import numpy as np
import pandas as pd

from rs_hmm import models
from rs_hmm.config import load_config
from rs_hmm.hmm import fit_macro_hmm
from rs_hmm.labels import make_labels
from rs_hmm.models import prepare_model_frame, train_all_models
from rs_hmm.simulation import run_simulation_from_config


def test_logistic_training_smoke() -> None:
    config = load_config("configs/sim_v1.yml")
    macro, loans, panel = run_simulation_from_config(config)
    labeled = make_labels(panel, config.label.horizon_months, config.behavior.dpd_default)
    macro_hmm, _, _ = fit_macro_hmm(macro)
    model_frame = prepare_model_frame(
        labeled=labeled,
        loans=loans,
        macro_hmm=macro_hmm,
        regime_threshold=config.evaluation.regime_threshold,
        horizon=config.label.horizon_months,
    )
    predictions, coefficients = train_all_models(
        model_frame,
        target=f"y_{config.label.horizon_months}m",
        train_frac=config.split.train_frac,
        val_frac=config.split.val_frac,
    )
    assert not predictions.empty
    assert not coefficients.empty


def _synthetic_empirical_frame(n_months: int = 20, loans_per_month: int = 10) -> pd.DataFrame:
    rows = []
    for month in range(n_months):
        p_stress = 0.08 + 0.84 * month / max(n_months - 1, 1)
        for loan_index in range(loans_per_month):
            recent_60 = int((month + loan_index) % 9 == 0)
            rows.append(
                {
                    "loan_id": f"loan_{loan_index:03d}",
                    "month": month,
                    "credit_score": 620 + ((loan_index * 13 + month) % 120),
                    "original_upb": 120_000 + loan_index * 2_500 + month * 100,
                    "original_interest_rate": 3.0 + ((loan_index + month) % 9) * 0.125,
                    "original_cltv": 55 + ((loan_index * 3 + month) % 35),
                    "original_dti": 20 + ((loan_index * 5 + month) % 25),
                    "loan_age_months": month,
                    "current_delinquency_months": recent_60,
                    "recent_60_dpd_3m": recent_60,
                    "months_since_30_dpd": (month + loan_index) % 18,
                    "recent_60_dpd_flag": recent_60,
                    "months_since_delinquency": (month + loan_index) % 18,
                    "rolling_6m_delinquency_rate": ((month + loan_index) % 4) / 6.0,
                    "custom_dpd_history_score": ((month * 2 + loan_index) % 5) / 5.0,
                    "inflation": 2.0 + 0.05 * month,
                    "policy_rate": 1.0 + 0.03 * month,
                    "unemployment": 4.0 + 0.02 * month,
                    "p_stress": p_stress,
                    "hard_stress": int(p_stress >= 0.5),
                    "p_stress_lag_3": max(p_stress - 0.05, 0.0),
                    "delta_p_stress": 0.02 + 0.002 * month,
                    "stress_duration": max(month - 6, 0),
                    "y_12m": int((month + loan_index) % 7 == 0 or (p_stress > 0.7 and loan_index % 5 == 0)),
                }
            )
    return pd.DataFrame(rows)


def test_empirical_specs_are_nested_and_use_imported_payment_history(monkeypatch) -> None:
    monkeypatch.setattr(
        models,
        "FREDDIE_NUMERIC_FEATURES",
        [
            "credit_score",
            "original_upb",
            "original_interest_rate",
            "original_cltv",
            "original_dti",
            "loan_age_months",
            "current_delinquency_months",
            "recent_60_dpd_3m",
            "months_since_30_dpd",
            "recent_60_dpd_flag",
            "months_since_delinquency",
            "rolling_6m_delinquency_rate",
            "custom_dpd_history_score",
        ],
    )
    frame = _synthetic_empirical_frame()

    specs = models._empirical_model_specs(frame)
    spec_by_name = {spec.name: spec for spec in specs}

    assert [spec.name for spec in specs] == [
        "baseline_logit",
        "macro_logit",
        "baseline_plus_p_stress_logit",
        "regime_aware_logit",
        "regime_aware_with_raw_macro_logit",
        "regime_aware_dynamic_stress_robustness_logit",
    ]
    assert "recent_60_dpd_flag" in spec_by_name["baseline_logit"].features
    assert "rolling_6m_delinquency_rate" in spec_by_name["macro_logit"].features
    assert "p_stress" not in spec_by_name["baseline_logit"].features
    assert "p_stress" in spec_by_name["baseline_plus_p_stress_logit"].features
    assert "inflation" not in spec_by_name["regime_aware_logit"].features
    assert "inflation" in spec_by_name["regime_aware_with_raw_macro_logit"].features
    assert "p_stress_lag_3" not in spec_by_name["regime_aware_dynamic_stress_robustness_logit"].features
    assert {"delta_p_stress", "stress_duration"}.issubset(
        spec_by_name["regime_aware_dynamic_stress_robustness_logit"].features
    )
    payment_interactions = [
        feature
        for feature in spec_by_name["regime_aware_logit"].interaction_base
        if feature
        in {
            "current_delinquency_months",
            "recent_60_dpd_3m",
            "months_since_30_dpd",
            "recent_60_dpd_flag",
            "months_since_delinquency",
            "rolling_6m_delinquency_rate",
            "custom_dpd_history_score",
        }
    ]
    assert payment_interactions == [
        "current_delinquency_months",
        "recent_60_dpd_3m",
        "months_since_30_dpd",
    ]


def test_empirical_training_writes_calibration_sensitivity_columns(monkeypatch) -> None:
    monkeypatch.setattr(
        models,
        "FREDDIE_NUMERIC_FEATURES",
        [
            "credit_score",
            "original_upb",
            "original_interest_rate",
            "original_cltv",
            "original_dti",
            "loan_age_months",
            "current_delinquency_months",
            "recent_60_dpd_3m",
            "months_since_30_dpd",
            "recent_60_dpd_flag",
            "months_since_delinquency",
            "rolling_6m_delinquency_rate",
        ],
    )
    frame = _synthetic_empirical_frame()

    predictions, coefficients = train_all_models(
        frame,
        target="y_12m",
        train_frac=0.6,
        val_frac=0.8,
    )

    expected_models = {
        "baseline_logit",
        "macro_logit",
        "baseline_plus_p_stress_logit",
        "regime_aware_logit",
        "regime_aware_with_raw_macro_logit",
        "regime_aware_dynamic_stress_robustness_logit",
    }
    assert set(predictions["model"]) == expected_models
    assert {
        "p",
        "p_raw",
        "p_platt",
        "p_intercept_only",
        "p_isotonic",
    }.issubset(predictions.columns)
    assert np.allclose(predictions["p"], predictions["p_platt"])
    for column in ["p", "p_raw", "p_platt", "p_intercept_only", "p_isotonic"]:
        assert predictions[column].between(1e-6, 1.0 - 1e-6).all()

    coefficient_features = set(coefficients["feature"])
    assert {
        "calibration_validation_brier_platt",
        "calibration_validation_brier_intercept_only",
        "calibration_validation_brier_isotonic",
        "calibration_validation_log_loss_platt",
        "calibration_validation_log_loss_intercept_only",
        "calibration_validation_log_loss_isotonic",
        "intercept_only_shift",
        "isotonic_threshold_count",
        "platt_logit_slope",
    }.issubset(coefficient_features)
