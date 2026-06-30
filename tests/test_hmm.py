from __future__ import annotations

import numpy as np
import pandas as pd

from rs_hmm.config import load_config
from rs_hmm.hmm import FACTOR_GROUPS, fit_macro_hmm
from rs_hmm.simulation import run_simulation_from_config


def test_hmm_outputs_posterior_columns() -> None:
    config = load_config("configs/sim_v1.yml")
    macro, _, _ = run_simulation_from_config(config)
    fitted, model, scaler = fit_macro_hmm(macro)
    assert {"p_stress", "hard_stress", "hmm_state"}.issubset(fitted.columns)
    assert fitted["p_stress"].between(0.0, 1.0).all()
    assert fitted["hmm_input_mode"].eq("raw").all()
    assert list(scaler.feature_names_in_) == ["inflation", "policy_rate"]
    assert model.means_.shape[1] == 2


def test_hmm_uses_interpretable_factors_for_expanded_macro_panel() -> None:
    rng = np.random.default_rng(42)
    n_months = 72
    month_index = np.arange(n_months, dtype=float)
    stress_step = (month_index >= n_months // 2).astype(float)

    macro = pd.DataFrame(
        {
            "month": np.arange(n_months),
            "month_date": pd.date_range("2019-01-01", periods=n_months, freq="MS"),
            "inflation": 2.0 + 1.0 * stress_step + rng.normal(0.0, 0.1, n_months),
            "policy_rate": 1.0 + 2.0 * stress_step + rng.normal(0.0, 0.1, n_months),
            "unemployment": 4.0 + 2.5 * stress_step + rng.normal(0.0, 0.1, n_months),
            "unemployment_change": 0.05 * stress_step + rng.normal(0.0, 0.03, n_months),
            "house_price_growth": 6.0 - 5.0 * stress_step + rng.normal(0.0, 0.1, n_months),
            "mortgage_rate": 3.5 + 1.4 * stress_step + rng.normal(0.0, 0.1, n_months),
            "mortgage_spread": 1.2 + 0.8 * stress_step + rng.normal(0.0, 0.05, n_months),
            "long_rate": 2.0 + 0.8 * stress_step + rng.normal(0.0, 0.08, n_months),
            "long_rate_change": 0.03 * stress_step + rng.normal(0.0, 0.03, n_months),
            "yield_curve": 1.0 - 0.9 * stress_step + rng.normal(0.0, 0.05, n_months),
            "initial_claims": 200_000.0 + 80_000.0 * stress_step + rng.normal(0.0, 2_000.0, n_months),
            "initial_claims_change": 800.0 * stress_step + rng.normal(0.0, 100.0, n_months),
            "payroll_growth": 1.8 - 1.2 * stress_step + rng.normal(0.0, 0.08, n_months),
            "real_income_growth": 2.0 - 1.0 * stress_step + rng.normal(0.0, 0.08, n_months),
            "financial_conditions": -0.4 + 0.9 * stress_step + rng.normal(0.0, 0.05, n_months),
            "consumer_sentiment": 95.0 - 20.0 * stress_step + rng.normal(0.0, 1.0, n_months),
            "consumer_sentiment_change": -0.5 * stress_step + rng.normal(0.0, 0.05, n_months),
        }
    )

    fitted, model, scaler = fit_macro_hmm(macro)

    factor_columns = list(FACTOR_GROUPS)
    assert set(factor_columns).issubset(fitted.columns)
    assert fitted["hmm_input_mode"].eq("interpretable_factors").all()
    assert fitted["hmm_input_columns"].eq(",".join(factor_columns)).all()
    assert list(scaler.feature_names_in_) == factor_columns
    assert model.means_.shape[1] == len(factor_columns)
