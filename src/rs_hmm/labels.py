from __future__ import annotations

import pandas as pd

from rs_hmm.empirical import canonicalize_panel


def make_labels(panel: pd.DataFrame, horizon: int, dpd_default: int) -> pd.DataFrame:
    panel = canonicalize_panel(panel).sort_values(["loan_id", "month"]).copy()

    default_times = (
        panel.loc[panel["dpd"] >= dpd_default]
        .groupby("loan_id")["month"]
        .min()
        .rename("T_default")
        .reset_index()
    )
    panel = panel.merge(default_times, on="loan_id", how="left")

    max_month = panel.groupby("loan_id")["month"].max().rename("max_month").reset_index()
    panel = panel.merge(max_month, on="loan_id", how="left")

    if "is_terminated" in panel.columns:
        terminal_times = (
            panel.loc[panel["is_terminated"].astype(bool)]
            .groupby("loan_id")["month"]
            .min()
            .rename("T_terminal")
            .reset_index()
        )
        panel = panel.merge(terminal_times, on="loan_id", how="left")
    else:
        panel["T_terminal"] = pd.NA

    panel["at_risk"] = (panel["dpd"] < dpd_default).astype(int)
    terminal_after_horizon = panel["T_terminal"].isna() | (panel["T_terminal"] > panel["month"] + horizon)
    panel["has_full_horizon"] = (
        (panel["month"] + horizon <= panel["max_month"]) & terminal_after_horizon
    ).astype(int)

    t = panel["month"]
    default_time = panel["T_default"]
    panel[f"y_{horizon}m"] = ((default_time > t) & (default_time <= t + horizon)).fillna(False).astype(int)

    return panel.loc[(panel["at_risk"] == 1) & (panel["has_full_horizon"] == 1)].copy()
