from __future__ import annotations

import numpy as np
import pandas as pd

from rs_hmm.freddie_mac import FREDDIE_MISSINGNESS_COLUMNS, FREDDIE_UNAVAILABLE_SENTINELS


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

FREDDIE_MISSINGNESS_FEATURES = list(FREDDIE_MISSINGNESS_COLUMNS.values())

FREDDIE_NUMERIC_FEATURES = (
    FREDDIE_STATIC_NUMERIC_FEATURES
    + FREDDIE_MISSINGNESS_FEATURES
    + FREDDIE_PAYMENT_HISTORY_FEATURES
)


def _month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _month_index(month_date: pd.Series) -> pd.Series:
    origin = month_date.min().to_period("M").ordinal
    return month_date.dt.to_period("M").map(lambda period: period.ordinal - origin).astype(int)


def reindex_internal_calendar_gaps(panel: pd.DataFrame) -> pd.DataFrame:
    """Insert explicit rows for missing calendar months inside each observed loan span."""
    required = {"loan_id", "month"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Cannot reindex servicing history; missing columns: {sorted(missing)}")

    out = panel.copy()
    out["month"] = pd.to_numeric(out["month"], errors="coerce")
    if out[["loan_id", "month"]].isna().any().any():
        raise ValueError("Calendar reindexing requires non-null loan_id and month values.")
    out["month"] = out["month"].astype(int)
    out = out.sort_values(["loan_id", "month"]).reset_index(drop=True)
    if out.duplicated(["loan_id", "month"]).any():
        raise ValueError("Calendar reindexing requires at most one servicing record per loan-month.")

    existing_observed = pd.to_numeric(out.get("servicing_record_observed", 1), errors="coerce")
    if not isinstance(existing_observed, pd.Series):
        existing_observed = pd.Series(existing_observed, index=out.index)
    out["servicing_record_observed"] = existing_observed.fillna(0).ne(0).astype("int8")
    out["servicing_gap"] = (out["servicing_record_observed"] == 0).astype("int8")

    previous_month = out.groupby("loan_id", sort=False)["month"].shift(1)
    gap_end_rows = out.loc[(out["month"] - previous_month) > 1]
    if gap_end_rows.empty:
        return out

    dynamic_columns = {
        "reporting_month",
        "monthly_reporting_period",
        "current_actual_upb",
        "current_loan_delinquency_status",
        "current_loan_delinquency_status_code",
        "loan_age_months",
        "remaining_months_to_legal_maturity",
        "defect_settlement_date",
        "modification_flag",
        "zero_balance_code",
        "zero_balance_effective_date",
        "zero_balance_effective_month",
        "termination_month_date",
        "termination_month",
        "current_interest_rate",
        "is_90_plus_dpd",
        "is_terminated",
        "dpd",
    }
    inserted: list[dict[str, object]] = []
    for index, current in gap_end_rows.iterrows():
        start = int(previous_month.loc[index]) + 1
        stop = int(current["month"])
        for missing_month in range(start, stop):
            row = current.to_dict()
            row["month"] = missing_month
            for column in dynamic_columns.intersection(row):
                row[column] = pd.NA
            if "month_date" in row and pd.notna(current.get("month_date")):
                row["month_date"] = pd.Timestamp(current["month_date"]) - pd.offsets.MonthBegin(
                    stop - missing_month
                )
            row["servicing_record_observed"] = 0
            row["servicing_gap"] = 1
            if "is_terminated" in out.columns:
                row["is_terminated"] = False
            inserted.append(row)

    if inserted:
        out = pd.concat([out, pd.DataFrame(inserted)], ignore_index=True, sort=False)
        out = out.sort_values(["loan_id", "month"]).reset_index(drop=True)
    return out


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

    out = reindex_internal_calendar_gaps(panel)
    out["dpd"] = pd.to_numeric(out["dpd"], errors="coerce").clip(lower=0)

    loan_id = out["loan_id"]
    month = out["month"]
    observed = out["servicing_record_observed"]
    out["current_delinquency_months"] = np.floor(out["dpd"] / 30.0).astype("Int16")
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

    position = out.groupby(loan_id, sort=False).cumcount() + 1
    observed_cumulative = observed.groupby(loan_id, sort=False).cumsum()
    for window in [3, 6, 12]:
        observed_in_window = observed_cumulative - observed_cumulative.groupby(
            loan_id, sort=False
        ).shift(window, fill_value=0)
        expected_in_window = position.clip(upper=window)
        out[f"history_{window}m_complete"] = observed_in_window.eq(expected_in_window).astype("int8")

    out["recent_30_dpd_3m"] = (out["delinquent_30_months_3m"] > 0).astype("int8")
    out["recent_60_dpd_3m"] = (out["delinquent_60_months_3m"] > 0).astype("int8")

    previous_30_dpd = out["current_30_dpd"].groupby(loan_id, sort=False).shift(1, fill_value=0)
    previous_observed = observed.groupby(loan_id, sort=False).shift(1, fill_value=0)
    out["cured_from_30_dpd"] = (
        (previous_observed == 1)
        & (observed == 1)
        & (previous_30_dpd == 1)
        & (out["current_30_dpd"] == 0)
    ).astype("int8")
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

    if "termination_month_date" in out.columns:
        termination_dates = _month_start(out["termination_month_date"])
        origin = out["month_date"].min().to_period("M").ordinal
        out["termination_month"] = termination_dates.dt.to_period("M").map(
            lambda period: period.ordinal - origin if pd.notna(period) else pd.NA
        )

    delinquency_months = pd.to_numeric(out["current_loan_delinquency_status"], errors="coerce")
    out["dpd"] = (delinquency_months.clip(lower=0) * 30).astype(float)
    out["is_terminated"] = out.get("is_terminated", False)

    for column in FREDDIE_STATIC_NUMERIC_FEATURES:
        if column in out.columns:
            numeric = pd.to_numeric(out[column], errors="coerce")
            sentinel = FREDDIE_UNAVAILABLE_SENTINELS.get(column)
            unavailable = numeric.eq(sentinel) if sentinel is not None else pd.Series(False, index=out.index)
            numeric = numeric.mask(unavailable)
            out[column] = numeric
            if column in FREDDIE_MISSINGNESS_COLUMNS:
                indicator = FREDDIE_MISSINGNESS_COLUMNS[column]
                existing = pd.to_numeric(out.get(indicator, 0), errors="coerce")
                if not isinstance(existing, pd.Series):
                    existing = pd.Series(existing, index=out.index)
                out[indicator] = (numeric.isna() | unavailable | existing.fillna(0).ne(0)).astype("int8")

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
