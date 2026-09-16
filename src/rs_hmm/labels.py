from __future__ import annotations

import pandas as pd

from rs_hmm.empirical import canonicalize_panel, reindex_internal_calendar_gaps


def make_labels(panel: pd.DataFrame, horizon: int, dpd_default: int) -> pd.DataFrame:
    panel = reindex_internal_calendar_gaps(canonicalize_panel(panel)).sort_values(
        ["loan_id", "month"]
    ).copy()

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
        terminal_time_column = "termination_month" if "termination_month" in panel.columns else "month"
        terminal_times = (
            panel.loc[panel["is_terminated"].astype(bool)]
            .groupby("loan_id")[terminal_time_column]
            .min()
            .rename("T_terminal")
            .reset_index()
        )
        panel = panel.merge(terminal_times, on="loan_id", how="left")
    else:
        panel["T_terminal"] = pd.NA

    observed = pd.to_numeric(panel.get("servicing_record_observed", 1), errors="coerce")
    if not isinstance(observed, pd.Series):
        observed = pd.Series(observed, index=panel.index)
    observed = observed.fillna(0).ne(0)
    before_first_default = panel["T_default"].isna() | (panel["month"] < panel["T_default"])
    complete_history = pd.to_numeric(
        panel.get("history_12m_complete", 1), errors="coerce"
    )
    if not isinstance(complete_history, pd.Series):
        complete_history = pd.Series(complete_history, index=panel.index)
    panel["at_risk"] = (
        observed & (panel["dpd"] < dpd_default) & before_first_default & complete_history.fillna(0).ne(0)
    ).astype(int)

    observed_cumulative = observed.astype(int).groupby(panel["loan_id"], sort=False).cumsum()
    observed_through_horizon = observed_cumulative.groupby(
        panel["loan_id"], sort=False
    ).shift(-horizon)
    future_observed_months = observed_through_horizon - observed_cumulative
    terminal_after_horizon = panel["T_terminal"].isna() | (panel["T_terminal"] > panel["month"] + horizon)
    panel["has_full_horizon"] = (
        future_observed_months.eq(horizon) & terminal_after_horizon
    ).astype(int)

    t = panel["month"]
    default_time = panel["T_default"]
    panel[f"y_{horizon}m"] = ((default_time > t) & (default_time <= t + horizon)).fillna(False).astype(int)

    return panel.loc[(panel["at_risk"] == 1) & (panel["has_full_horizon"] == 1)].copy()
