from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    macro = pd.read_csv("data/processed/macro_monthly.csv")
    panel = pd.read_csv("data/processed/loan_monthly_panel.csv")
    labeled = pd.read_csv("data/processed/loan_monthly_panel_labeled.csv")

    os.makedirs("reports/figures", exist_ok=True)

    # 1) Regime plot
    fig, ax = plt.subplots()
    ax.plot(macro["month"], macro["regime_true"])
    ax.set_title("True regime over time (0=normal, 1=inflation-stress)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Regime")
    fig.savefig("reports/figures/01_regime_true.png", dpi=200, bbox_inches="tight")

    # 2) Macro series
    fig, ax = plt.subplots()
    ax.plot(macro["month"], macro["inflation"], label="inflation")
    ax.plot(macro["month"], macro["policy_rate"], label="policy_rate")
    ax.set_title("Macro series over time")
    ax.set_xlabel("Month")
    ax.legend()
    fig.savefig("reports/figures/02_macro_series.png", dpi=200, bbox_inches="tight")

    # 3) Portfolio default rate over time (from panel)
    # Default event = first time dpd>=90 for a loan. Compute event month and event rate.
    default_events = (
        panel.loc[panel["dpd"] >= 90]
        .groupby("loan_id")["month"]
        .min()
        .reset_index()
        .rename(columns={"month": "event_month"})
    )
    event_counts = (
        default_events.groupby("event_month").size().rename("n_events").reset_index()
    )
    event_counts["event_rate"] = event_counts["n_events"] / panel["loan_id"].nunique()

    fig, ax = plt.subplots()
    ax.plot(event_counts["event_month"], event_counts["event_rate"])
    ax.set_title("Default event rate by month (first 90+ DPD)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Event rate")
    fig.savefig(
        "reports/figures/03_default_event_rate.png", dpi=200, bbox_inches="tight"
    )

    # 4) DPD trajectories (sample a few loans)
    sample_ids = panel["loan_id"].drop_duplicates().sample(6, random_state=1).tolist()
    fig, ax = plt.subplots()
    for lid in sample_ids:
        s = panel.loc[panel["loan_id"] == lid].sort_values("month")
        ax.plot(s["month"], s["dpd"], label=f"loan {lid}")
    ax.set_title("Sample DPD trajectories")
    ax.set_xlabel("Month")
    ax.set_ylabel("DPD")
    ax.legend(fontsize=7)
    fig.savefig("reports/figures/04_dpd_trajectories.png", dpi=200, bbox_inches="tight")

    # Tables: default rates by regime (using labeled rows)
    has_regime = "regime_true" in labeled.columns
    regime_all_nan = has_regime and labeled["regime_true"].isna().all()
    if has_regime and not regime_all_nan:
        merged = labeled
    elif has_regime and regime_all_nan:
        merged = labeled.merge(
            macro[["month", "regime_true"]],
            on="month",
            how="left",
            suffixes=("_labeled", "_macro"),
        )
        merged["regime_true"] = merged["regime_true_macro"]
    else:
        merged = labeled.merge(macro[["month", "regime_true"]], on="month", how="left")
    summary = merged.groupby("regime_true")["y_6m"].agg(["mean", "count"]).reset_index()
    summary.to_csv("reports/figures/summary_y6m_by_regime.csv", index=False)

    print("Saved figures to reports/figures/ and summary_y6m_by_regime.csv")


if __name__ == "__main__":
    main()
