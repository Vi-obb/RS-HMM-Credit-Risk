from __future__ import annotations

import pandas as pd

from rs_hmm.labels import make_labels


def test_make_labels_applies_horizon_and_censoring() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1, 1, 1],
            "month": [0, 1, 2, 3],
            "dpd": [0, 0, 90, 90],
        }
    )
    labeled = make_labels(panel, horizon=2, dpd_default=90)
    assert labeled["y_2m"].tolist() == [1, 1]
    assert labeled["month"].tolist() == [0, 1]


def test_make_labels_requires_every_future_calendar_month() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1, 1, 1, 1],
            "month": [0, 1, 3, 4, 5],
            "dpd": [0, 0, 0, 0, 0],
        }
    )

    labeled = make_labels(panel, horizon=2, dpd_default=90)

    assert labeled["month"].tolist() == [3]
    assert labeled["has_full_horizon"].eq(1).all()


def test_make_labels_does_not_allow_post_default_reentry() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": [1] * 7,
            "month": list(range(7)),
            "dpd": [0, 0, 90, 0, 0, 0, 0],
        }
    )

    labeled = make_labels(panel, horizon=2, dpd_default=90)

    assert labeled["month"].tolist() == [0, 1]
    assert labeled["y_2m"].tolist() == [1, 1]


def test_make_labels_uses_effective_termination_month() -> None:
    panel = pd.DataFrame(
        {
            "loan_id": [1, 1, 1, 1],
            "month": [0, 1, 2, 3],
            "dpd": [0, 0, 0, 0],
            "is_terminated": [False, False, True, False],
            "termination_month": [pd.NA, pd.NA, 3, pd.NA],
        }
    )

    labeled = make_labels(panel, horizon=2, dpd_default=90)

    assert labeled["month"].tolist() == [0]
