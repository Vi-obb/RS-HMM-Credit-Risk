from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd


FRED_SERIES = (
    "CPIAUCSL",
    "FEDFUNDS",
    "UNRATE",
    "HPIPONM226S",
    "MORTGAGE30US",
    "GS10",
    "T10Y2Y",
    "ICSA",
    "PAYEMS",
    "DSPIC96",
    "NFCI",
    "UMCSENT",
)
FRED_GRAPH_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={','.join(FRED_SERIES)}"
BASE_REQUIRED_SERIES = {"observation_date", "CPIAUCSL", "FEDFUNDS", "UNRATE"}
TRANSFORMED_MACRO_COLUMNS = [
    "inflation",
    "policy_rate",
    "unemployment",
    "unemployment_change",
    "house_price_growth",
    "mortgage_rate",
    "mortgage_spread",
    "long_rate",
    "long_rate_change",
    "yield_curve",
    "initial_claims",
    "initial_claims_change",
    "payroll_growth",
    "real_income_growth",
    "financial_conditions",
    "consumer_sentiment",
    "consumer_sentiment_change",
]


def _month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()


def read_fred_graph(url: str = FRED_GRAPH_URL) -> pd.DataFrame:
    with urlopen(url, timeout=30) as response:
        payload = response.read()

    stream = BytesIO(payload)
    try:
        with ZipFile(stream) as archive:
            frames = [
                pd.read_csv(archive.open(name))
                for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]
    except BadZipFile:
        stream.seek(0)
        return pd.read_csv(stream)

    if not frames:
        raise ValueError("FRED graph ZIP did not contain any CSV files.")
    return pd.concat(frames, ignore_index=True, sort=False)


def _monthly_average(raw: pd.DataFrame) -> pd.DataFrame:
    series_columns = [column for column in FRED_SERIES if column in raw.columns]
    out = raw[["observation_date", *series_columns]].copy()
    out["month_date"] = _month_start(out["observation_date"])
    for column in series_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    monthly = (
        out.dropna(subset=["month_date"])
        .groupby("month_date", as_index=False)[series_columns]
        .mean()
        .sort_values("month_date")
        .reset_index(drop=True)
    )
    if monthly.empty:
        return monthly

    full_months = pd.date_range(monthly["month_date"].min(), monthly["month_date"].max(), freq="MS")
    return monthly.set_index("month_date").reindex(full_months).rename_axis("month_date").reset_index()


def _add_column(out: pd.DataFrame, column: str, values: pd.Series, feature_columns: list[str]) -> None:
    out[column] = values
    feature_columns.append(column)


def build_macro_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize public FRED macro series into monthly mortgage-PD model inputs."""
    missing = BASE_REQUIRED_SERIES.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing FRED columns: {sorted(missing)}")

    monthly = _monthly_average(raw)
    feature_columns: list[str] = []

    _add_column(monthly, "inflation", monthly["CPIAUCSL"].pct_change(12) * 100.0, feature_columns)
    _add_column(monthly, "policy_rate", monthly["FEDFUNDS"], feature_columns)
    _add_column(monthly, "unemployment", monthly["UNRATE"], feature_columns)
    _add_column(monthly, "unemployment_change", monthly["UNRATE"].diff(), feature_columns)

    if "HPIPONM226S" in monthly.columns:
        _add_column(
            monthly,
            "house_price_growth",
            monthly["HPIPONM226S"].pct_change(12) * 100.0,
            feature_columns,
        )
    if "MORTGAGE30US" in monthly.columns:
        _add_column(monthly, "mortgage_rate", monthly["MORTGAGE30US"], feature_columns)
    if {"MORTGAGE30US", "GS10"}.issubset(monthly.columns):
        _add_column(
            monthly,
            "mortgage_spread",
            monthly["MORTGAGE30US"] - monthly["GS10"],
            feature_columns,
        )
    if "GS10" in monthly.columns:
        _add_column(monthly, "long_rate", monthly["GS10"], feature_columns)
        _add_column(monthly, "long_rate_change", monthly["GS10"].diff(), feature_columns)
    if "T10Y2Y" in monthly.columns:
        _add_column(monthly, "yield_curve", monthly["T10Y2Y"], feature_columns)
    if "ICSA" in monthly.columns:
        _add_column(monthly, "initial_claims", monthly["ICSA"], feature_columns)
        _add_column(monthly, "initial_claims_change", monthly["ICSA"].diff(), feature_columns)
    if "PAYEMS" in monthly.columns:
        _add_column(monthly, "payroll_growth", monthly["PAYEMS"].pct_change(12) * 100.0, feature_columns)
    if "DSPIC96" in monthly.columns:
        _add_column(monthly, "real_income_growth", monthly["DSPIC96"].pct_change(12) * 100.0, feature_columns)
    if "NFCI" in monthly.columns:
        _add_column(monthly, "financial_conditions", monthly["NFCI"], feature_columns)
    if "UMCSENT" in monthly.columns:
        _add_column(monthly, "consumer_sentiment", monthly["UMCSENT"], feature_columns)
        _add_column(monthly, "consumer_sentiment_change", monthly["UMCSENT"].diff(), feature_columns)

    required_features = ["inflation", "policy_rate", "unemployment"]
    monthly = monthly.dropna(subset=["month_date", *required_features]).copy()
    origin = monthly["month_date"].min().to_period("M").ordinal
    monthly["month"] = monthly["month_date"].dt.to_period("M").map(lambda period: period.ordinal - origin)
    columns = ["month", "month_date"] + [column for column in TRANSFORMED_MACRO_COLUMNS if column in monthly.columns]
    return monthly[columns].reset_index(drop=True)


def fetch_fred_macro(output_path: str | Path, start: str | None = None, end: str | None = None) -> Path:
    raw = read_fred_graph(FRED_GRAPH_URL)
    macro = build_macro_monthly(raw)
    if start is not None:
        macro = macro[macro["month_date"] >= pd.Timestamp(start)]
    if end is not None:
        macro = macro[macro["month_date"] <= pd.Timestamp(end)]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(output, index=False)
    return output
