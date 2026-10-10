"""Information-timed rerun utilities; historical freeze outputs remain untouched."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import json
import time

import numpy as np
import pandas as pd

from rs_hmm.empirical import canonicalize_panel, reindex_internal_calendar_gaps, FREDDIE_NUMERIC_FEATURES
from rs_hmm.macro import FRED_SERIES, TRANSFORMED_MACRO_COLUMNS, build_macro_monthly
from rs_hmm.workflow import EMPIRICAL_PANEL_COLUMNS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observable_labels(raw: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """First 90+ DPD before voluntary payoff; unresolved exits remain censored.

    Require continuous servicing to the first observed default, voluntary payoff,
    or horizon end, rather than requiring survival for the whole horizon.
    """
    panel = reindex_internal_calendar_gaps(canonicalize_panel(raw)).sort_values(["loan_id", "month"]).reset_index(drop=True)
    observed = panel["servicing_record_observed"].eq(1)
    status_known = observed & panel["dpd"].notna()
    payoff_record = observed & panel["is_terminated"].fillna(False) & pd.to_numeric(panel["zero_balance_code"], errors="coerce").eq(1)
    defaults = panel.loc[observed & panel["dpd"].ge(90)].groupby("loan_id")["month"].min()
    terminal_rows = panel.loc[panel["is_terminated"].fillna(False)].copy()
    terminals = terminal_rows.groupby("loan_id")["termination_month"].min()
    codes = pd.to_numeric(terminal_rows["zero_balance_code"], errors="coerce")
    payoffs = terminal_rows.loc[codes.eq(1)].groupby("loan_id")["termination_month"].min()
    default = panel["loan_id"].map(defaults)
    terminal = pd.to_numeric(panel["loan_id"].map(terminals), errors="coerce")
    payoff = pd.to_numeric(panel["loan_id"].map(payoffs), errors="coerce")
    month = panel["month"]
    horizon_end = month + horizon
    # A default at the exit month takes precedence; later defaults are unavailable.
    positive = default.gt(month) & default.le(horizon_end) & (terminal.isna() | default.le(terminal))
    payoff_negative = payoff.gt(month) & payoff.le(horizon_end) & payoff.eq(terminal) & ~positive
    resolution = horizon_end.where(~positive, default).where(~payoff_negative, payoff)
    cumulative = (status_known | payoff_record).astype("int32").groupby(panel["loan_id"], sort=False).cumsum()
    lookup = pd.Series(cumulative.to_numpy(), index=pd.MultiIndex.from_frame(panel[["loan_id", "month"]]))
    keys = pd.MultiIndex.from_arrays([panel["loan_id"], resolution])
    through = lookup.reindex(keys).to_numpy()
    continuous = (through - cumulative.to_numpy()) == (resolution - month).to_numpy()
    no_unknown_exit = terminal.isna() | terminal.gt(horizon_end) | positive | payoff_negative
    status_cumulative = status_known.astype("int32").groupby(panel["loan_id"], sort=False).cumsum()
    lookback_known = status_cumulative - status_cumulative.groupby(panel["loan_id"], sort=False).shift(12, fill_value=0)
    expected_lookback = (panel.groupby("loan_id", sort=False).cumcount() + 1).clip(upper=12)
    at_risk = (status_known & panel["dpd"].lt(90) & (default.isna() | month.lt(default))
               & (terminal.isna() | month.lt(terminal)) & panel["history_12m_complete"].eq(1)
               & lookback_known.eq(expected_lookback))
    panel["y_12m"] = positive.astype("int8")
    panel["outcome_observed"] = continuous & no_unknown_exit
    panel["at_risk"] = at_risk.astype("int8")
    panel["outcome_resolution"] = np.select([positive, payoff_negative], ["default", "voluntary_payoff"], default="full_horizon")
    # Uniform horizon maturity avoids selectively admitting early-known positives.
    panel["label_available_date"] = panel["month_date"] + pd.DateOffset(months=horizon + 2)
    panel["forecast_issue_date"] = panel["month_date"] + pd.DateOffset(months=1)
    return panel


def build_observable_panel(source: Path, output: Path, audit_path: Path, observation_end: str = "2024-03-01") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: list[dict[str, object]] = []
    carry = pd.DataFrame()
    last_loan = None
    columns = EMPIRICAL_PANEL_COLUMNS | {"zero_balance_code"}
    reader = pd.read_csv(source, usecols=lambda name: name in columns,
                         dtype={"loan_sequence_number": "str", "zero_balance_code": "str"}, chunksize=250_000)
    first = True
    def emit(raw: pd.DataFrame) -> None:
        nonlocal first, last_loan
        if raw.empty:
            return
        ids = raw["loan_sequence_number"]
        if not ids.is_monotonic_increasing or (last_loan is not None and ids.iloc[0] <= last_loan):
            raise ValueError("Streaming labels require source ordered by loan; a loan cannot recur across batches")
        last_loan = ids.iloc[-1]
        labeled = observable_labels(raw)
        # Administrative test cut: every scored horizon must mature within the source window.
        keep = labeled["at_risk"].eq(1) & labeled["outcome_observed"] & labeled["month_date"].le(observation_end)
        for date, group in labeled.groupby("month_date", sort=True):
            risk = group["at_risk"].eq(1)
            known = risk & group["outcome_observed"]
            counts.append({"month_date": date, "at_risk": int(risk.sum()), "known_outcome": int(known.sum()),
                           "censored": int((risk & ~group["outcome_observed"]).sum()),
                           "positive_labels": int(group.loc[known, "y_12m"].sum()),
                           "payoff_negatives": int((known & group["outcome_resolution"].eq("voluntary_payoff")).sum())})
        result = labeled.loc[keep, ["loan_id", "month_date", "y_12m", "label_available_date", "forecast_issue_date", "outcome_resolution", *FREDDIE_NUMERIC_FEATURES]]
        result.to_csv(output, mode="w" if first else "a", header=first, index=False)
        first = False
        print(f"Labels: through {last_loan}; batch {len(raw):,} source / {len(result):,} eligible", flush=True)
    for chunk in reader:
        combined = pd.concat([carry, chunk], ignore_index=True)
        final_id = combined["loan_sequence_number"].iloc[-1]
        carry = combined.loc[combined["loan_sequence_number"].eq(final_id)].copy()
        emit(combined.loc[combined["loan_sequence_number"].ne(final_id)])
    emit(carry)
    pd.DataFrame(counts).groupby("month_date", as_index=False).sum(numeric_only=True).to_csv(audit_path, index=False)


def read_vintage(payload: bytes, vintage: pd.Timestamp) -> pd.DataFrame:
    frames = []
    streams = [payload]
    if payload[:2] == b"PK":
        with ZipFile(BytesIO(payload)) as archive:
            streams = [archive.read(name) for name in archive.namelist() if name.endswith(".csv")]
    suffix = "_" + vintage.strftime("%Y%m%d")
    for data in streams:
        frame = pd.read_csv(BytesIO(data))
        for name in frame.columns:
            if name != "observation_date" and not name.endswith(suffix):
                raise ValueError(f"ALFRED returned wrong vintage column: {name}, expected {suffix}")
        frame = frame.rename(columns={name: name.removesuffix(suffix) for name in frame.columns})
        frames.append(frame)
    raw = frames[0]
    for frame in frames[1:]:
        raw = raw.merge(frame, on="observation_date", how="outer", validate="one_to_one")
    if not set(FRED_SERIES).issubset(raw):
        raise ValueError("ALFRED response missing a required source series")
    return raw.loc[pd.to_datetime(raw["observation_date"]).le(vintage)].sort_values("observation_date")


def build_realtime_macro(output: Path, cache: Path, start="2014-01-01", end="2024-03-01") -> None:
    cache.mkdir(parents=True, exist_ok=True)
    manifest = []
    def fetch(month: pd.Timestamp) -> dict[str, object]:
        issue = month + pd.DateOffset(months=1)
        vintage = issue - pd.Timedelta(days=1)
        query = urlencode({"id": ",".join(FRED_SERIES), "vintage_date": ",".join([vintage.strftime("%Y-%m-%d")] * len(FRED_SERIES))})
        url = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?" + query
        path = cache / f"{vintage:%Y-%m-%d}.bin"
        if not path.exists():
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=60) as response:
                        payload = response.read()
                    read_vintage(payload, vintage)
                    path.write_bytes(payload)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
        raw = read_vintage(path.read_bytes(), vintage)
        transformed = build_macro_monthly(raw)
        complete = transformed.loc[transformed["month_date"].le(month)].dropna(subset=TRANSFORMED_MACRO_COLUMNS)
        if complete.empty:
            raise ValueError(f"No complete available macro vector for {month}")
        row = complete.iloc[-1].to_dict()
        source = row["month_date"]
        row.update(month_date=month, source_macro_month=source, vintage_date=vintage, forecast_issue_date=issue)
        manifest.append({"month_date": str(month.date()), "vintage_date": str(vintage.date()), "url": url, "sha256": sha256(path), "source_macro_month": str(source.date())})
        print(f"Macro: {month:%Y-%m} as-of {vintage:%Y-%m-%d}, complete vector {source:%Y-%m}", flush=True)
        return row
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(fetch, pd.date_range(start, end, freq="MS")))
    pd.DataFrame(rows).sort_values("month_date").to_csv(output, index=False)
    (cache / "manifest.json").write_text(json.dumps(sorted(manifest, key=lambda row: row["month_date"]), indent=2))
