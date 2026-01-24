from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_regimes(
    n_months: int, A: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Simulate a latent Markov regime sequence z_t."""
    n_regimes = A.shape[0]
    z = np.zeros(n_months, dtype=int)
    for t in range(1, n_months):
        z[t] = rng.choice(n_regimes, p=A[z[t - 1]])
    return z


def simulate_macro(
    z: np.ndarray,
    infl_mean: np.ndarray,
    infl_sd: np.ndarray,
    policy_base: float,
    policy_beta_infl: float,
    policy_noise_sd: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate macro series conditional on regime."""
    n_months = len(z)
    inflation = rng.normal(loc=infl_mean[z], scale=infl_sd[z], size=n_months)
    # Keep inflation non-negative and reasonable
    inflation = np.clip(inflation, 0.0, 0.6)

    policy_rate = (
        policy_base
        + policy_beta_infl * inflation
        + rng.normal(0.0, policy_noise_sd, size=n_months)
    )
    policy_rate = np.clip(policy_rate, 0.0, 0.8)

    df = pd.DataFrame(
        {"month": np.arange(n_months), "regime_true": z, "inflation": inflation, "policy_rate": policy_rate}
    )

    # NEW: stress duration (months since entering stress, 0 in normal)
    stress_dur = np.zeros(n_months, dtype=int)
    for t in range(1, n_months):
        if df.loc[t, "regime_true"] == 1:
            stress_dur[t] = stress_dur[t - 1] + 1 if df.loc[t - 1, "regime_true"] == 1 else 1
        else:
            stress_dur[t] = 0
    df["stress_duration"] = stress_dur

    return df



def simulate_loans_static(
    n_loans: int,
    n_months: int,
    principal_mean: float,
    principal_sd: float,
    tenor_min: int,
    tenor_max: int,
    base_interest_rate: float,
    beta_borrower_risk_rate: float,
    beta_stress_rate: float,
    macro: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate loan origination properties and borrower latent risk."""
    borrower_risk = rng.normal(0.0, 1.0, size=n_loans)

    orig_month = rng.integers(low=0, high=max(1, n_months - tenor_min), size=n_loans)
    tenor = rng.integers(low=tenor_min, high=tenor_max + 1, size=n_loans)

    principal = rng.normal(principal_mean, principal_sd, size=n_loans)
    principal = np.clip(principal, 300.0, None)

    # Interest rate depends on borrower risk + stress at origination (optional)
    stress_at_orig = macro.loc[orig_month, "regime_true"].to_numpy()
    interest_rate = (
        base_interest_rate
        + beta_borrower_risk_rate * borrower_risk
        + beta_stress_rate * stress_at_orig
    )
    interest_rate = np.clip(interest_rate, 0.05, 0.9)

    return pd.DataFrame(
        {
            "loan_id": np.arange(n_loans),
            "borrower_id": np.arange(n_loans),  # keep 1-to-1 for MVP
            "orig_month": orig_month,
            "tenor_months": tenor,
            "principal": principal,
            "interest_rate": interest_rate,
            "borrower_risk": borrower_risk,
        }
    )


def simulate_panel(
    loans: pd.DataFrame,
    macro: pd.DataFrame,
    miss_alpha: float,
    miss_beta_u: float,
    miss_beta_infl: float,
    miss_beta_policy: float,
    miss_gamma_regime: float,
    miss_delta_duration: float,
    dpd_step: int,
    dpd_cure_step: int,
    dpd_default: int,
    dpd_cap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate monthly loan panel with DPD dynamics and payment behavior."""
    rows = []
    for _, r in loans.iterrows():
        loan_id = int(r["loan_id"])
        u = float(r["borrower_risk"])
        start = int(r["orig_month"])
        tenor = int(r["tenor_months"])
        end = min(start + tenor, int(macro["month"].max()) + 1)

        dpd = 0
        defaulted = False
        default_month = None

        # Simple scheduled payment proxy (not amortized; just constant fraction)
        scheduled_payment = float(r["principal"]) / max(tenor, 1)

        for t in range(start, end):
            infl = float(macro.loc[t, "inflation"])
            pol = float(macro.loc[t, "policy_rate"])
            reg = int(macro.loc[t, "regime_true"])
            dur = float(macro.loc[t, "stress_duration"])


            if defaulted:
                # After default, keep loan observable but in default status (for MVP)
                rows.append(
                    {
                        "loan_id": loan_id,
                        "month": t,
                        "dpd": dpd,
                        "is_active": 1,
                        "defaulted": 1,
                        "default_month": default_month,
                        "scheduled_payment": scheduled_payment,
                        "paid_amount": 0.0,
                        "missed_payment": 1,
                        "inflation": infl,
                        "policy_rate": pol,
                        "regime_true": reg,
                        "stress_duration": dur,
                    }
                )
                continue

            logit_p = (
                miss_alpha
                + miss_beta_u * u
                + miss_beta_infl * infl
                + miss_beta_policy * pol
                + miss_gamma_regime * reg
                + miss_delta_duration * np.log1p(dur)
            )


            p_miss = float(sigmoid(np.array([logit_p]))[0])
            miss = int(rng.uniform() < p_miss)

            if miss == 1:
                paid = 0.0
                dpd = min(dpd + dpd_step, dpd_cap)
            else:
                paid = scheduled_payment
                dpd = max(dpd - dpd_cure_step, 0)

            # default trigger
            if dpd >= dpd_default:
                defaulted = True
                default_month = t

            rows.append(
                {
                    "loan_id": loan_id,
                    "month": t,
                    "dpd": dpd,
                    "is_active": 1,
                    "defaulted": int(defaulted),
                    "default_month": default_month,
                    "scheduled_payment": scheduled_payment,
                    "paid_amount": paid,
                    "missed_payment": miss,
                    "inflation": infl,
                    "policy_rate": pol,
                    "stress_duration": dur,
                }
            )

    return pd.DataFrame(rows)


def make_labels(panel: pd.DataFrame, horizon: int, dpd_default: int) -> pd.DataFrame:
    """
    Create y_{i,t}^{(h)} = 1 if loan hits default (dpd>=dpd_default) within next h months.
    Conservative: requires loan-month to have full horizon of observability in panel.
    """
    panel = panel.sort_values(["loan_id", "month"]).copy()

    # Compute T_i: first month dpd>=default threshold
    default_times = (
        panel.loc[panel["dpd"] >= dpd_default]
        .groupby("loan_id")["month"]
        .min()
        .rename("T_default")
        .reset_index()
    )
    panel = panel.merge(default_times, on="loan_id", how="left")

    # At-risk: dpd < default at t, not already defaulted at t
    panel["at_risk"] = (panel["dpd"] < dpd_default).astype(int)

    # Need full horizon: check that (t+6) exists for that loan in panel
    # We do this by computing max observed month per loan
    max_month = (
        panel.groupby("loan_id")["month"].max().rename("max_month").reset_index()
    )
    panel = panel.merge(max_month, on="loan_id", how="left")
    panel["has_full_horizon"] = (panel["month"] + horizon <= panel["max_month"]).astype(
        int
    )

    # Label: default occurs in (t, t+h]
    # If T_default is NaN => never defaults in observed window -> label 0
    t = panel["month"]
    T = panel["T_default"]
    panel["y_6m"] = ((T > t) & (T <= t + horizon)).fillna(False).astype(int)

    # Apply censoring rule: keep only at-risk and full horizon
    panel = panel.loc[(panel["at_risk"] == 1) & (panel["has_full_horizon"] == 1)].copy()

    return panel
