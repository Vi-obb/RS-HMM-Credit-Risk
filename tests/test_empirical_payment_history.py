from __future__ import annotations

import pandas as pd

from rs_hmm.empirical import add_payment_history_features, canonicalize_panel


def test_add_payment_history_features_uses_only_current_and_past_months() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": ["A", "A", "A", "A", "A", "A", "A"],
            "month": [3, 0, 6, 1, 4, 2, 5],
            "dpd": [60, 0, 90, 30, 0, 0, 0],
        }
    )

    features = add_payment_history_features(panel)
    loan_a = features.loc[features["loan_id"] == "A"].set_index("month")

    assert features[["loan_id", "month"]].to_dict("records") == [
        {"loan_id": "A", "month": 0},
        {"loan_id": "A", "month": 1},
        {"loan_id": "A", "month": 2},
        {"loan_id": "A", "month": 3},
        {"loan_id": "A", "month": 4},
        {"loan_id": "A", "month": 5},
        {"loan_id": "A", "month": 6},
    ]

    assert loan_a.loc[0, "recent_30_dpd_3m"] == 0
    assert loan_a.loc[0, "delinquent_30_months_3m"] == 0
    assert loan_a.loc[0, "months_since_30_dpd"] == 1

    assert loan_a.loc[1, "current_delinquency_months"] == 1
    assert loan_a.loc[1, "current_30_dpd"] == 1
    assert loan_a.loc[1, "months_since_30_dpd"] == 0

    assert loan_a.loc[2, "cured_from_30_dpd"] == 1
    assert loan_a.loc[2, "cumulative_30_dpd_cures"] == 1
    assert loan_a.loc[2, "months_since_30_dpd_cure"] == 0

    assert loan_a.loc[3, "current_delinquency_months"] == 2
    assert loan_a.loc[3, "current_60_dpd"] == 1
    assert loan_a.loc[3, "months_since_60_dpd"] == 0
    assert loan_a.loc[3, "delinquent_30_months_3m"] == 2
    assert loan_a.loc[3, "delinquent_60_months_3m"] == 1

    assert loan_a.loc[5, "delinquent_30_months_6m"] == 2
    assert loan_a.loc[5, "months_since_30_dpd"] == 2
    assert loan_a.loc[5, "months_since_30_dpd_cure"] == 1

    assert loan_a.loc[6, "current_delinquency_months"] == 3
    assert loan_a.loc[6, "delinquent_30_months_6m"] == 3
    assert loan_a.loc[6, "delinquent_30_months_12m"] == 3
    assert loan_a.loc[6, "recent_60_dpd_3m"] == 1


def test_payment_history_features_are_loan_local() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": ["A", "A", "B", "B"],
            "month": [0, 1, 0, 1],
            "dpd": [30, 0, 0, 0],
        }
    )

    features = add_payment_history_features(panel)
    loan_b = features.loc[features["loan_id"] == "B"].set_index("month")

    assert loan_b.loc[0, "recent_30_dpd_3m"] == 0
    assert loan_b.loc[1, "recent_30_dpd_3m"] == 0
    assert loan_b.loc[1, "months_since_30_dpd"] == 2
    assert loan_b.loc[1, "cumulative_30_dpd_cures"] == 0


def test_canonicalize_panel_preserves_synthetic_path_behavior() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1],
            "month": [0, 1],
            "dpd": [0, 30],
        }
    )

    canonical = canonicalize_panel(panel)

    assert canonical.equals(panel)
