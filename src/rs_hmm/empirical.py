from __future__ import annotations

import numpy as np
import pandas as pd


FREDDIE_NUMERIC_FEATURES = [
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


def _month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _month_index(month_date: pd.Series) -> pd.Series:
    origin = month_date.min().to_period("M").ordinal
    return month_date.dt.to_period("M").map(lambda period: period.ordinal - origin).astype(int)


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

    for column in FREDDIE_NUMERIC_FEATURES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["current_actual_upb"] = pd.to_numeric(out.get("current_actual_upb", np.nan), errors="coerce")
    if "original_upb" in out.columns:
        out["current_actual_upb"] = out["current_actual_upb"].fillna(out["original_upb"])
    if "current_interest_rate" in out.columns and "original_interest_rate" in out.columns:
        out["current_interest_rate"] = out["current_interest_rate"].fillna(out["original_interest_rate"])

    return out.sort_values(["loan_id", "month"]).reset_index(drop=True)


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
