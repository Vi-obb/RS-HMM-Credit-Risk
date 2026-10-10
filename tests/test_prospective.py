from __future__ import annotations

import pandas as pd
import pytest

from rs_hmm.prospective import observable_labels, read_vintage
from rs_hmm.rolling_freeze import RollingFold


def panel(months, dpd, code="", exit_month=None):
    n = len(months)
    return pd.DataFrame({"loan_id": ["a"] * n, "month": months, "dpd": dpd,
                         "month_date": [pd.Timestamp("2020-01-01") + pd.DateOffset(months=m) for m in months],
                         "is_terminated": [m == exit_month for m in months],
                         "termination_month": [exit_month if m == exit_month else pd.NA for m in months],
                         "zero_balance_code": [code if m == exit_month else "" for m in months],
                         "history_12m_complete": [1] * n})


def test_early_default_does_not_require_survival_to_horizon():
    result = observable_labels(panel([0, 1, 2], [0, 30, 90]), horizon=12)
    eligible = result.loc[result.at_risk.eq(1) & result.outcome_observed]
    assert eligible.month.tolist() == [0, 1]
    assert eligible.y_12m.tolist() == [1, 1]
    assert eligible.label_available_date.iloc[0] == pd.Timestamp("2021-03-01")


def test_payoff_resolves_negative_but_sale_remains_censored():
    payoff = observable_labels(panel([0, 1, 2], [0, 0, 0], "01", 2))
    sale = observable_labels(panel([0, 1, 2], [0, 0, 0], "15", 2))
    assert payoff.loc[:1, "outcome_observed"].all()
    assert payoff.loc[:1, "y_12m"].eq(0).all()
    assert not sale.outcome_observed.any()
    assert payoff.at_risk.iloc[2] == 0


def test_gap_before_default_does_not_resolve_unknown_outcome():
    result = observable_labels(panel([0, 2], [0, 90]))
    assert not result.loc[result.month.eq(0), "outcome_observed"].iloc[0]


def test_default_on_payoff_month_takes_precedence():
    result = observable_labels(panel([0, 1, 2], [0, 30, 90], "01", 2))
    assert result.loc[:1, "y_12m"].eq(1).all()
    assert result.loc[:1, "outcome_observed"].all()


def test_default_after_sale_is_not_known_before_exit():
    result = observable_labels(panel([0, 1, 2], [0, 0, 90], "15", 1))
    assert not result.outcome_observed.iloc[0]


def test_fully_observed_negative_and_missing_horizon():
    result = observable_labels(panel([0, 1, 2, 3], [0, 0, 0, 0]), horizon=2)
    assert result.outcome_observed.tolist() == [True, True, False, False]


def test_wrong_vintage_is_rejected():
    with pytest.raises(ValueError, match="wrong vintage"):
        read_vintage(b"observation_date,UNRATE_20220101\n2020-01-01,3.5\n", pd.Timestamp("2021-01-01"))


def test_corrected_fold_windows_disjoint_and_mature():
    from rs_hmm.rolling_freeze import PROSPECTIVE_FOLDS
    for fold in PROSPECTIVE_FOLDS:
        assert pd.Timestamp(fold.train_end) < pd.Timestamp(fold.validation_start)
        assert pd.Timestamp(fold.validation_end) < pd.Timestamp(fold.calibration_start)
        assert pd.Timestamp(fold.calibration_end) + pd.DateOffset(months=14) < pd.Timestamp(fold.test_start) + pd.DateOffset(months=2)
