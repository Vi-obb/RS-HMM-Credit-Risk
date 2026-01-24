from __future__ import annotations
import os
import yaml
import numpy as np
import pandas as pd

from rs_hmm.simulate_panel import (
    simulate_regimes,
    simulate_macro,
    simulate_loans_static,
    simulate_panel,
    make_labels,
)


def main():
    with open("configs/sim_v1.yml", "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["simulation"]["seed"]
    rng = np.random.default_rng(seed)

    n_months = cfg["simulation"]["n_months"]
    n_loans = cfg["simulation"]["n_loans"]

    A = np.array(cfg["regimes"]["transition_matrix"], dtype=float)

    # Regimes + Macro
    z = simulate_regimes(n_months, A, rng)
    infl_mean = np.array(cfg["macro"]["inflation"]["mean"], dtype=float)
    infl_sd = np.array(cfg["macro"]["inflation"]["sd"], dtype=float)

    macro = simulate_macro(
        z=z,
        infl_mean=infl_mean,
        infl_sd=infl_sd,
        policy_base=float(cfg["macro"]["policy_rate"]["base"]),
        policy_beta_infl=float(cfg["macro"]["policy_rate"]["beta_inflation"]),
        policy_noise_sd=float(cfg["macro"]["policy_rate"]["noise_sd"]),
        rng=rng,
    )

    # Loans
    loans = simulate_loans_static(
        n_loans=n_loans,
        n_months=n_months,
        principal_mean=float(cfg["loans"]["principal_mean"]),
        principal_sd=float(cfg["loans"]["principal_sd"]),
        tenor_min=int(cfg["loans"]["tenor_min"]),
        tenor_max=int(cfg["loans"]["tenor_max"]),
        base_interest_rate=float(cfg["loans"]["base_interest_rate"]),
        beta_borrower_risk_rate=float(cfg["loans"]["beta_borrower_risk_rate"]),
        beta_stress_rate=float(cfg["loans"]["beta_stress_rate"]),
        macro=macro,
        rng=rng,
    )

    # Panel
    panel = simulate_panel(
        loans=loans,
        macro=macro,
        miss_alpha=float(cfg["behavior"]["miss_logit"]["alpha"]),
        miss_beta_u=float(cfg["behavior"]["miss_logit"]["beta_u"]),
        miss_beta_infl=float(cfg["behavior"]["miss_logit"]["beta_inflation"]),
        miss_beta_policy=float(cfg["behavior"]["miss_logit"]["beta_policy"]),
        miss_gamma_regime=float(cfg["behavior"]["miss_logit"]["gamma_regime"]),
        miss_delta_duration=float(cfg["behavior"]["miss_logit"]["delta_duration"]),
        dpd_step=int(cfg["behavior"]["dpd_step"]),
        dpd_cure_step=int(cfg["behavior"]["dpd_cure_step"]),
        dpd_default=int(cfg["behavior"]["dpd_default"]),
        dpd_cap=int(cfg["behavior"]["dpd_cap"]),
        rng=rng,
    )

    # Labels (6-month horizon)
    horizon = int(cfg["label"]["horizon_months"])
    dpd_default = int(cfg["behavior"]["dpd_default"])
    panel_labeled = make_labels(panel, horizon=horizon, dpd_default=dpd_default)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    macro.to_csv("data/processed/macro_monthly.csv", index=False)
    loans.to_csv("data/processed/loans_static.csv", index=False)
    panel.to_csv("data/processed/loan_monthly_panel.csv", index=False)
    panel_labeled.to_csv("data/processed/loan_monthly_panel_labeled.csv", index=False)

    # Quick summaries
    print("Saved:")
    print(" - data/processed/macro_monthly.csv")
    print(" - data/processed/loans_static.csv")
    print(" - data/processed/loan_monthly_panel.csv")
    print(" - data/processed/loan_monthly_panel_labeled.csv")
    print()
    print("Panel rows:", len(panel), " | Labeled rows:", len(panel_labeled))
    print("Event rate (y_6m):", panel_labeled["y_6m"].mean())


if __name__ == "__main__":
    main()
