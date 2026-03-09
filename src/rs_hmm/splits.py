from __future__ import annotations

import numpy as np
import pandas as pd


def time_split_by_month(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.85
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = np.sort(df["month"].unique())
    train_cutoff = months[int(len(months) * train_frac)]
    val_cutoff = months[int(len(months) * val_frac)]
    train = df[df["month"] < train_cutoff].copy()
    val = df[(df["month"] >= train_cutoff) & (df["month"] < val_cutoff)].copy()
    test = df[df["month"] >= val_cutoff].copy()
    return train, val, test
