from __future__ import annotations

import pandas as pd

from rs_hmm.splits import time_split_by_month


def test_time_split_is_chronological() -> None:
    df = pd.DataFrame({"month": list(range(10)), "value": list(range(10))})
    train, val, test = time_split_by_month(df, train_frac=0.6, val_frac=0.8)
    assert train["month"].max() < val["month"].min()
    assert val["month"].max() < test["month"].min()
