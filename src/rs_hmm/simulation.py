from __future__ import annotations

import numpy as np
import pandas as pd

from rs_hmm.config import AppConfig


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_regimes(
    n_months: int, transition_matrix: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n_regimes = transition_matrix.shape[0]
    z = np.zeros(n_months, dtype=int)
    for month in range(1, n_months):
        z[month] = rng.choice(n_regimes, p=transition_matrix[z[month - 1]])
    return z


def simulate_macro(config: AppConfig, regimes: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    inflation = rng.normal(
        loc=np.asarray(config.macro.inflation.mean)[regimes],
        scale=np.asarray(config.macro.inflation.sd)[regimes],
        size=len(regimes),
    )
    inflation = np.clip(inflation, 0.0, 0.6)

    policy_rate = (
        config.macro.policy_rate.base
        + config.macro.policy_rate.beta_inflation * inflation
        + rng.normal(0.0, config.macro.policy_rate.noise_sd, size=len(regimes))
    )
    policy_rate = np.clip(policy_rate, 0.0, 0.8)

    stress_duration = np.zeros(len(regimes), dtype=int)
    for month in range(1, len(regimes)):
        if regimes[month] == 1:
            stress_duration[month] = (
                stress_duration[month - 1] + 1 if regimes[month - 1] == 1 else 1
            )

    return pd.DataFrame(
        {
            "month": np.arange(len(regimes)),
            "regime_true": regimes,
            "inflation": inflation,
            "policy_rate": policy_rate,
            "stress_duration": stress_duration,
        }
    )


def simulate_loans_static(config: AppConfig, macro: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n_loans = config.simulation.n_loans
    borrower_risk = rng.normal(0.0, 1.0, size=n_loans)
    orig_month = rng.integers(
        low=0,
        high=max(1, config.simulation.n_months - config.loans.tenor_min),
        size=n_loans,
    )
    tenor_months = rng.integers(
        low=config.loans.tenor_min,
        high=config.loans.tenor_max + 1,
        size=n_loans,
    )
    principal = np.clip(
        rng.normal(config.loans.principal_mean, config.loans.principal_sd, size=n_loans),
        300.0,
        None,
    )
    stress_at_orig = macro.loc[orig_month, "regime_true"].to_numpy()
    interest_rate = (
        config.loans.base_interest_rate
        + config.loans.beta_borrower_risk_rate * borrower_risk
        + config.loans.beta_stress_rate * stress_at_orig
    )
    interest_rate = np.clip(interest_rate, 0.05, 0.9)

    return pd.DataFrame(
        {
            "loan_id": np.arange(n_loans),
            "borrower_id": np.arange(n_loans),
            "orig_month": orig_month,
            "tenor_months": tenor_months,
            "principal": principal,
            "interest_rate": interest_rate,
            "borrower_risk": borrower_risk,
        }
    )


def simulate_panel(
    config: AppConfig,
    loans: pd.DataFrame,
    macro: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict] = []
    miss = config.behavior.miss_logit
    for _, loan in loans.iterrows():
        loan_id = int(loan["loan_id"])
        borrower_risk = float(loan["borrower_risk"])
        start = int(loan["orig_month"])
        tenor = int(loan["tenor_months"])
        end = min(start + tenor, int(macro["month"].max()) + 1)
        scheduled_payment = float(loan["principal"]) / max(tenor, 1)

        dpd = 0
        defaulted = False
        default_month = np.nan

        for month in range(start, end):
            macro_row = macro.loc[month]
            inflation = float(macro_row["inflation"])
            policy_rate = float(macro_row["policy_rate"])
            regime_true = int(macro_row["regime_true"])
            stress_duration = int(macro_row["stress_duration"])

            if defaulted:
                rows.append(
                    {
                        "loan_id": loan_id,
                        "month": month,
                        "dpd": dpd,
                        "is_active": 1,
                        "defaulted": 1,
                        "default_month": default_month,
                        "scheduled_payment": scheduled_payment,
                        "paid_amount": 0.0,
                        "missed_payment": 1,
                        "inflation": inflation,
                        "policy_rate": policy_rate,
                        "regime_true": regime_true,
                        "stress_duration": stress_duration,
                    }
                )
                continue

            logit_p = (
                miss.alpha
                + miss.beta_u * borrower_risk
                + miss.beta_inflation * inflation
                + miss.beta_policy * policy_rate
                + miss.gamma_regime * regime_true
                + miss.delta_duration * np.log1p(stress_duration)
            )
            p_miss = float(sigmoid(np.asarray([logit_p]))[0])
            missed_payment = int(rng.uniform() < p_miss)

            if missed_payment:
                paid_amount = 0.0
                dpd = min(dpd + config.behavior.dpd_step, config.behavior.dpd_cap)
            else:
                paid_amount = scheduled_payment
                dpd = max(dpd - config.behavior.dpd_cure_step, 0)

            if dpd >= config.behavior.dpd_default:
                defaulted = True
                default_month = month

            rows.append(
                {
                    "loan_id": loan_id,
                    "month": month,
                    "dpd": dpd,
                    "is_active": 1,
                    "defaulted": int(defaulted),
                    "default_month": default_month,
                    "scheduled_payment": scheduled_payment,
                    "paid_amount": paid_amount,
                    "missed_payment": missed_payment,
                    "inflation": inflation,
                    "policy_rate": policy_rate,
                    "regime_true": regime_true,
                    "stress_duration": stress_duration,
                }
            )

    return pd.DataFrame(rows)


def run_simulation_from_config(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.simulation.seed)
    transition_matrix = np.asarray(config.regimes.transition_matrix, dtype=float)
    regimes = simulate_regimes(config.simulation.n_months, transition_matrix, rng)
    macro = simulate_macro(config, regimes, rng)
    loans = simulate_loans_static(config, macro, rng)
    panel = simulate_panel(config, loans, macro, rng)
    return macro, loans, panel
