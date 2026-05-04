from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import re

import pandas as pd


ORIGINATION_COLUMNS = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer_flag",
    "maturity_date",
    "msa_md",
    "mi_percent",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "prepayment_penalty_flag",
    "product_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_sequence_number",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "servicer_name",
    "super_conforming_flag",
    "pre_relief_refinance_loan_sequence_number",
    "special_eligibility_program",
    "relief_refinance_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "mi_cancellation_indicator",
]

PERFORMANCE_COLUMNS = [
    "loan_sequence_number",
    "monthly_reporting_period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age_months",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
]


@dataclass(frozen=True)
class FreddieMacIngestionResult:
    origination_path: Path
    performance_path: Path
    panel_path: Path


def _read_pipe_file(
    path: Path,
    columns: list[str],
    usecols: list[int] | range | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    positions = list(range(32)) if usecols is None else list(usecols)
    raw_columns = [f"field_{index:02d}" for index in range(1, len(positions) + 1)]
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            if len(row) < len(positions):
                row = row + [""] * (len(positions) - len(row))
            rows.append([row[position] if position < len(row) else "" for position in positions])

    frame = pd.DataFrame(rows, columns=raw_columns)
    out = frame.copy()
    for index, column in enumerate(columns, start=1):
        out[column] = out[f"field_{index:02d}"]
    return out


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", pd.NA), errors="coerce")


def _parse_period(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.replace("", pd.NA), format="%Y%m", errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp()


def _parse_delinquency_status(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"^(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def load_origination_file(path: Path, cohort_year: int, max_rows: int | None = None) -> pd.DataFrame:
    frame = _read_pipe_file(path, ORIGINATION_COLUMNS, max_rows=max_rows)
    frame["cohort_year"] = cohort_year
    frame["loan_sequence_number"] = frame["loan_sequence_number"].astype(str)
    frame["first_payment_month"] = _parse_period(frame["first_payment_date"])
    frame["maturity_month"] = _parse_period(frame["maturity_date"])
    for column in [
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
        "property_valuation_method",
    ]:
        frame[column] = _to_numeric(frame[column])
    return frame


def load_performance_file(path: Path, cohort_year: int, max_rows: int | None = None) -> pd.DataFrame:
    frame = _read_pipe_file(path, PERFORMANCE_COLUMNS, usecols=range(11), max_rows=max_rows)
    frame["cohort_year"] = cohort_year
    frame["loan_sequence_number"] = frame["loan_sequence_number"].astype(str)
    frame["reporting_month"] = _parse_period(frame["monthly_reporting_period"])
    frame["current_actual_upb"] = _to_numeric(frame["current_actual_upb"])
    frame["current_interest_rate"] = _to_numeric(frame["current_interest_rate"])
    frame["loan_age_months"] = _to_numeric(frame["loan_age_months"])
    frame["remaining_months_to_legal_maturity"] = _to_numeric(frame["remaining_months_to_legal_maturity"])
    frame["current_loan_delinquency_status_code"] = frame["current_loan_delinquency_status"].astype(str)
    frame["current_loan_delinquency_status"] = _parse_delinquency_status(frame["current_loan_delinquency_status"])
    return frame


def build_loan_month_panel(origination: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    panel = performance.merge(
        origination,
        on=["loan_sequence_number", "cohort_year"],
        how="left",
        suffixes=("_perf", "_orig"),
    )
    panel["is_90_plus_dpd"] = (panel["current_loan_delinquency_status"] >= 3).astype("Int64")
    panel["is_terminated"] = panel["zero_balance_code"].astype(str).ne("")
    panel = panel.sort_values(["loan_sequence_number", "reporting_month"]).reset_index(drop=True)
    return panel


def discover_sample_files(data_root: Path, years: list[int] | None = None) -> list[tuple[int, Path, Path]]:
    sample_root = data_root / "Freddie Mac Mortgage Data"
    if years is None:
        years = list(range(2015, 2026))

    discovered: list[tuple[int, Path, Path]] = []
    for year in years:
        year_dir = sample_root / f"sample_{year}"
        orig_path = year_dir / f"sample_orig_{year}.txt"
        perf_path = year_dir / f"sample_svcg_{year}.txt"
        if not orig_path.exists():
            raise FileNotFoundError(orig_path)
        if not perf_path.exists():
            raise FileNotFoundError(perf_path)
        discovered.append((year, orig_path, perf_path))
    return discovered


def run_freddie_ingestion(
    data_root: str | Path,
    output_dir: str | Path,
    years: list[int] | None = None,
    max_rows: int | None = None,
) -> FreddieMacIngestionResult:
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    origination_frames: list[pd.DataFrame] = []
    performance_frames: list[pd.DataFrame] = []

    for year, orig_path, perf_path in discover_sample_files(data_root, years=years):
        origination_frames.append(load_origination_file(orig_path, year, max_rows=max_rows))
        performance_frames.append(load_performance_file(perf_path, year, max_rows=max_rows))

    origination = pd.concat(origination_frames, ignore_index=True)
    performance = pd.concat(performance_frames, ignore_index=True)
    panel = build_loan_month_panel(origination, performance)

    origination_path = output_dir / "freddie_origination_clean.csv"
    performance_path = output_dir / "freddie_performance_clean.csv"
    panel_path = output_dir / "freddie_loan_month_panel.csv"

    origination.to_csv(origination_path, index=False)
    performance.to_csv(performance_path, index=False)
    panel.to_csv(panel_path, index=False)

    return FreddieMacIngestionResult(
        origination_path=origination_path,
        performance_path=performance_path,
        panel_path=panel_path,
    )
