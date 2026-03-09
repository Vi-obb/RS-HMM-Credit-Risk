from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def compute_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split), frame in predictions.groupby(["model", "split"], sort=True):
        y = frame["y"].to_numpy()
        p = frame["p"].to_numpy()
        rows.append(
            {
                "model": model,
                "split": split,
                "n": int(len(frame)),
                "event_rate": float(np.mean(y)),
                "pred_mean": float(np.mean(p)),
                "auc": safe_auc(y, p),
                "brier": float(brier_score_loss(y, p)),
                "logloss": float(log_loss(y, p, labels=[0, 1])),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "brier", "model"]).reset_index(drop=True)


def compute_regime_slice_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = predictions[predictions["split"] == "test"].copy()
    for model, frame in test.groupby("model", sort=True):
        slices = {
            "overall": frame,
            "normal": frame[frame["hard_stress"] == 0],
            "stress": frame[frame["hard_stress"] == 1],
        }
        for slice_name, slice_df in slices.items():
            if slice_df.empty:
                continue
            y = slice_df["y"].to_numpy()
            p = slice_df["p"].to_numpy()
            rows.append(
                {
                    "model": model,
                    "slice": slice_name,
                    "n": int(len(slice_df)),
                    "event_rate": float(np.mean(y)),
                    "pred_mean": float(np.mean(p)),
                    "auc": safe_auc(y, p),
                    "brier": float(brier_score_loss(y, p)),
                    "logloss": float(log_loss(y, p, labels=[0, 1])),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "slice"]).reset_index(drop=True)


def compute_time_bucket_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return pd.DataFrame()

    test["time_bucket"] = (test["month"] // 12).astype(int)
    rows = []
    for (model, time_bucket), frame in test.groupby(["model", "time_bucket"], sort=True):
        y = frame["y"].to_numpy()
        p = frame["p"].to_numpy()
        rows.append(
            {
                "model": model,
                "time_bucket": int(time_bucket),
                "n": int(len(frame)),
                "event_rate": float(np.mean(y)),
                "pred_mean": float(np.mean(p)),
                "brier": float(brier_score_loss(y, p)),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "time_bucket"]).reset_index(drop=True)


def calibration_points(frame: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    return calibration_curve(frame["y"], frame["p"], n_bins=n_bins, strategy="quantile")
