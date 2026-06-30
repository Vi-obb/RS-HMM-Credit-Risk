from __future__ import annotations

import numpy as np
import pandas as pd


FREDDIE_STATIC_NUMERIC_FEATURES = [
    "credit_score",
    "mi_percent",
    "number_of_units",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
    "loan_age_months",
    "remaining_months_to_legal_maturity",
    "current_actual_upb",
    "current_interest_rate",
]

FREDDIE_PAYMENT_HISTORY_FEATURES = [
    "current_delinquency_months",
    "current_30_dpd",
    "current_60_dpd",
    "recent_30_dpd_3m",
    "recent_60_dpd_3m",
    "ever_30_dpd",
    "ever_60_dpd",
    "months_since_30_dpd",
    "months_since_60_dpd",
    "delinquent_30_months_3m",
    "delinquent_30_months_6m",
    "delinquent_30_months_12m",
    "delinquent_60_months_3m",
    "delinquent_60_months_6m",
    "delinquent_60_months_12m",
    "cured_from_30_dpd",
    "cumulative_30_dpd_cures",
    "months_since_30_dpd_cure",
]

FREDDIE_NUMERIC_FEATURES = FREDDIE_STATIC_NUMERIC_FEATURES + FREDDIE_PAYMENT_HISTORY_FEATURES


def _month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _month_index(month_date: pd.Series) -> pd.Series:
    origin = month_date.min().to_period("M").ordinal
    return month_date.dt.to_period("M").map(lambda period: period.ordinal - origin).astype(int)


def _months_since_event(month: pd.Series, event: pd.Series, loan_id: pd.Series) -> pd.Series:
    last_event_month = month.where(event.eq(1)).groupby(loan_id, sort=False).ffill()
    first_month = month.groupby(loan_id, sort=False).transform("min")
    return (month - last_event_month).where(last_event_month.notna(), month - first_month + 1)


def _rolling_event_count(event: pd.Series, loan_id: pd.Series, window: int) -> pd.Series:
    cumulative = event.groupby(loan_id, sort=False).cumsum()
    lagged = cumulative.groupby(loan_id, sort=False).shift(window, fill_value=0)
    return cumulative - lagged


def add_payment_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add Freddie payment-history features using information available through month t."""
    required = {"loan_id", "month", "dpd"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Cannot build payment-history features; missing columns: {sorted(missing)}")

    out = panel.copy()
    out["month"] = pd.to_numeric(out["month"], errors="coerce")
    if out[["loan_id", "month"]].isna().any().any():
        raise ValueError("Payment-history features require non-null loan_id and month values.")

    out["dpd"] = pd.to_numeric(out["dpd"], errors="coerce").fillna(0).clip(lower=0)
    out = out.sort_values(["loan_id", "month"]).reset_index(drop=True)

    loan_id = out["loan_id"]
    month = out["month"]
    out["current_delinquency_months"] = np.floor(out["dpd"] / 30.0).astype("int16")
    out["current_30_dpd"] = (out["dpd"] >= 30).astype("int8")
    out["current_60_dpd"] = (out["dpd"] >= 60).astype("int8")

    out["ever_30_dpd"] = out["current_30_dpd"].groupby(loan_id, sort=False).cummax().astype("int8")
    out["ever_60_dpd"] = out["current_60_dpd"].groupby(loan_id, sort=False).cummax().astype("int8")
    out["months_since_30_dpd"] = _months_since_event(month, out["current_30_dpd"], loan_id).astype("float32")
    out["months_since_60_dpd"] = _months_since_event(month, out["current_60_dpd"], loan_id).astype("float32")

    for threshold, source in [("30", out["current_30_dpd"]), ("60", out["current_60_dpd"])]:
        for window in [3, 6, 12]:
            out[f"delinquent_{threshold}_months_{window}m"] = (
                _rolling_event_count(source, loan_id, window).astype("int16")
            )

    out["recent_30_dpd_3m"] = (out["delinquent_30_months_3m"] > 0).astype("int8")
    out["recent_60_dpd_3m"] = (out["delinquent_60_months_3m"] > 0).astype("int8")

    previous_30_dpd = out["current_30_dpd"].groupby(loan_id, sort=False).shift(1, fill_value=0)
    out["cured_from_30_dpd"] = ((previous_30_dpd == 1) & (out["current_30_dpd"] == 0)).astype("int8")
    out["cumulative_30_dpd_cures"] = (
        out["cured_from_30_dpd"].groupby(loan_id, sort=False).cumsum().astype("int16")
    )
    out["months_since_30_dpd_cure"] = _months_since_event(
        month,
        out["cured_from_30_dpd"],
        loan_id,
    ).astype("float32")

    return out


def canonicalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready loan-month panel from either synthetic or Freddie rows."""
    if {"loan_id", "month", "dpd"}.issubset(panel.columns):
        out = panel.copy()
        if "month_date" in out.columns:
            out["month_date"] = _month_start(out["month_date"])
        return out

    required = {
        "loan_sequence_number",
        "reporting_month",
        "current_loan_delinquency_status",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Cannot canonicalize panel; missing columns: {sorted(missing)}")

    out = panel.copy()
    out["loan_id"] = out["loan_sequence_number"].astype(str)
    out["month_date"] = _month_start(out["reporting_month"])
    out = out.dropna(subset=["loan_id", "month_date"]).copy()
    out["month"] = _month_index(out["month_date"])

    delinquency_months = pd.to_numeric(out["current_loan_delinquency_status"], errors="coerce").fillna(0)
    out["dpd"] = (delinquency_months.clip(lower=0) * 30).astype(float)
    out["is_terminated"] = out.get("is_terminated", False)

    for column in FREDDIE_STATIC_NUMERIC_FEATURES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["current_actual_upb"] = pd.to_numeric(out.get("current_actual_upb", np.nan), errors="coerce")
    if "original_upb" in out.columns:
        out["current_actual_upb"] = out["current_actual_upb"].fillna(out["original_upb"])
    if "current_interest_rate" in out.columns and "original_interest_rate" in out.columns:
        out["current_interest_rate"] = out["current_interest_rate"].fillna(out["original_interest_rate"])

    return add_payment_history_features(out)


def align_macro_months(macro: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if "month_date" not in macro.columns:
        return macro
    out = macro.copy()
    out["month_date"] = _month_start(out["month_date"])
    if "month_date" not in panel.columns:
        return out
    panel_months = _month_start(panel["month_date"])
    origin = panel_months.min().to_period("M").ordinal
    out["month"] = out["month_date"].dt.to_period("M").map(lambda period: period.ordinal - origin).astype(int)
    return out
