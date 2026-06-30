from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from rs_hmm import macro as macro_module
from rs_hmm.macro import FRED_GRAPH_URL, FRED_SERIES, build_macro_monthly, read_fred_graph


def _expanded_fred_raw(n_months: int = 18) -> pd.DataFrame:
    month_index = np.arange(n_months, dtype=float)
    return pd.DataFrame(
        {
            "observation_date": pd.date_range("2020-01-01", periods=n_months, freq="MS"),
            "CPIAUCSL": 100.0 + month_index,
            "FEDFUNDS": 1.0 + 0.1 * month_index,
            "UNRATE": 4.0 + 0.2 * month_index,
            "HPIPONM226S": 200.0 + 2.0 * month_index,
            "MORTGAGE30US": 3.0 + 0.1 * month_index,
            "GS10": 2.0 + 0.05 * month_index,
            "T10Y2Y": 1.5 - 0.02 * month_index,
            "ICSA": 200_000.0 + 1_000.0 * month_index,
            "PAYEMS": 150_000.0 + 1_500.0 * month_index,
            "DSPIC96": 16_000.0 + 100.0 * month_index,
            "NFCI": -0.5 + 0.05 * month_index,
            "UMCSENT": 90.0 - 0.5 * month_index,
        }
    )


def test_fred_endpoint_includes_expanded_public_candidate_set() -> None:
    assert FRED_GRAPH_URL.endswith(f"id={','.join(FRED_SERIES)}")
    assert FRED_SERIES == (
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


def test_build_macro_monthly_normalizes_expanded_series() -> None:
    raw = _expanded_fred_raw()
    extra_weekly_row = raw.iloc[[12]].copy()
    extra_weekly_row["observation_date"] = pd.Timestamp("2021-01-15")
    extra_weekly_row.loc[:, [column for column in FRED_SERIES if column != "MORTGAGE30US"]] = np.nan
    extra_weekly_row["MORTGAGE30US"] = 6.0

    macro = build_macro_monthly(pd.concat([raw, extra_weekly_row], ignore_index=True))

    expected_columns = {
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
    }
    assert expected_columns.issubset(macro.columns)
    assert macro["month_date"].iloc[0] == pd.Timestamp("2021-01-01")
    assert macro["month"].iloc[0] == 0

    first = macro.iloc[0]
    assert first["inflation"] == pytest.approx(12.0)
    assert first["house_price_growth"] == pytest.approx(12.0)
    assert first["unemployment_change"] == pytest.approx(0.2)
    assert first["mortgage_rate"] == pytest.approx(5.1)
    assert first["mortgage_spread"] == pytest.approx(5.1 - 2.6)
    assert first["long_rate_change"] == pytest.approx(0.05)
    assert first["initial_claims_change"] == pytest.approx(1_000.0)
    assert first["payroll_growth"] == pytest.approx(12.0)
    assert first["real_income_growth"] == pytest.approx(7.5)
    assert first["consumer_sentiment_change"] == pytest.approx(-0.5)


def test_build_macro_monthly_accepts_original_three_series() -> None:
    raw = _expanded_fred_raw()[["observation_date", "CPIAUCSL", "FEDFUNDS", "UNRATE"]]

    macro = build_macro_monthly(raw)

    assert {"inflation", "policy_rate", "unemployment"}.issubset(macro.columns)
    assert "mortgage_rate" not in macro.columns
    assert "mortgage_spread" not in macro.columns


def test_read_fred_graph_accepts_zip_members(monkeypatch) -> None:
    payload = BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("README.txt", "metadata")
        archive.writestr("monthly.csv", "observation_date,CPIAUCSL,FEDFUNDS,UNRATE\n2020-01-01,100,1,4\n")
        archive.writestr("weekly,_ending_thursday.csv", "observation_date,MORTGAGE30US\n2020-01-02,3.5\n")

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return payload.getvalue()

    monkeypatch.setattr(macro_module, "urlopen", lambda _url, timeout: Response())

    raw = read_fred_graph("https://example.test/fredgraph.csv")

    assert {"CPIAUCSL", "FEDFUNDS", "UNRATE", "MORTGAGE30US"}.issubset(raw.columns)
    assert len(raw) == 2
