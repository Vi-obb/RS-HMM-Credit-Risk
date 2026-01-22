from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


def main():
    macro = pd.read_csv("data/processed/macro_monthly.csv")

    # HMM observations
    obs_cols = ["inflation", "policy_rate"]
    X = macro[obs_cols].values

    # Standardize (helps HMM numerical stability)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Fit 2-state Gaussian HMM
    from hmmlearn.hmm import GaussianHMM

    hmm = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=500,
        random_state=42,
    )
    hmm.fit(Xs)

    # Posterior probabilities for each state at each time
    post = hmm.predict_proba(Xs)  # shape (T, 2)

    # Identify which hidden state corresponds to "stress"
    # We'll choose the state with higher mean inflation (in original units)
    means_std = hmm.means_  # in standardized space
    means_orig = scaler.inverse_transform(means_std)
    state_infl_means = means_orig[:, 0]  # inflation is col 0

    stress_state = int(np.argmax(state_infl_means))
    p_stress = post[:, stress_state]

    out = macro[["month", "regime_true", "inflation", "policy_rate"]].copy()
    out["p_stress"] = p_stress
    out["hmm_state"] = np.argmax(post, axis=1)
    out["hmm_stress_state"] = stress_state

    os.makedirs("data/processed", exist_ok=True)
    out.to_csv("data/processed/macro_with_hmm.csv", index=False)

    # Diagnostics plots
    os.makedirs("reports/figures", exist_ok=True)

    # 1) p_stress over time vs true regime
    fig, ax = plt.subplots()
    ax.plot(out["month"], out["p_stress"], label="p_stress (HMM)")
    ax.plot(out["month"], out["regime_true"], label="true_regime", linestyle="--")
    ax.set_title("HMM stress probability vs true regime")
    ax.set_xlabel("Month")
    ax.legend()
    fig.savefig(
        "reports/figures/hmm_p_stress_vs_true.png", dpi=200, bbox_inches="tight"
    )

    # 2) confusion / accuracy (simulation-only)
    pred_regime = (out["p_stress"] >= 0.5).astype(int)
    acc = (pred_regime == out["regime_true"]).mean()
    print("Stress-state chosen:", stress_state, " | inflation means:", state_infl_means)
    print("Regime classification accuracy (p_stress>=0.5):", float(acc))

    print("Saved:")
    print(" - data/processed/macro_with_hmm.csv")
    print(" - reports/figures/hmm_p_stress_vs_true.png")


if __name__ == "__main__":
    main()
