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
