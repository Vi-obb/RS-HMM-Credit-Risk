from __future__ import annotations

from pathlib import Path

import pandas as pd


FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL,FEDFUNDS,UNRATE"


def _month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def build_macro_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize FRED CPI, Fed Funds, and unemployment into model inputs."""
    required = {"observation_date", "CPIAUCSL", "FEDFUNDS", "UNRATE"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing FRED columns: {sorted(missing)}")

    out = raw.copy()
    out["month_date"] = _month_start(out["observation_date"])
    for column in ["CPIAUCSL", "FEDFUNDS", "UNRATE"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.sort_values("month_date").reset_index(drop=True)
    out["inflation"] = out["CPIAUCSL"].pct_change(12) * 100.0
    out["policy_rate"] = out["FEDFUNDS"]
    out["unemployment"] = out["UNRATE"]
    out = out.dropna(subset=["month_date", "inflation", "policy_rate", "unemployment"]).copy()
    origin = out["month_date"].min().to_period("M").ordinal
    out["month"] = out["month_date"].dt.to_period("M").map(lambda period: period.ordinal - origin)
    return out[["month", "month_date", "inflation", "policy_rate", "unemployment"]].reset_index(drop=True)


def fetch_fred_macro(output_path: str | Path, start: str | None = None, end: str | None = None) -> Path:
    raw = pd.read_csv(FRED_GRAPH_URL)
    macro = build_macro_monthly(raw)
    if start is not None:
        macro = macro[macro["month_date"] >= pd.Timestamp(start)]
    if end is not None:
        macro = macro[macro["month_date"] <= pd.Timestamp(end)]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(output, index=False)
    return output
