from __future__ import annotations

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve


def safe_auc(y, p):
    # AUC undefined if only one class in slice
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def summarize_slice(df: pd.DataFrame, name: str, slice_name: str):
    y = df["y"].astype(int).to_numpy()
    p = df["p"].astype(float).to_numpy()
    return {
        "model": name,
        "slice": slice_name,
        "n": len(df),
        "event_rate": float(np.mean(y)) if len(df) else np.nan,
        "pred_mean": float(np.mean(p)) if len(df) else np.nan,
        "auc": safe_auc(y, p) if len(df) else np.nan,
        "brier": brier_score_loss(y, p) if len(df) else np.nan,
        "logloss": log_loss(y, p) if len(df) else np.nan,
    }


def plot_calibration(df: pd.DataFrame, title: str, outpath: str):
    y = df["y"].astype(int).to_numpy()
    p = df["p"].astype(float).to_numpy()

    # If too few positives in a slice, calibration curve can get noisy
    if len(df) < 300:
        return

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots()
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default frequency")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    pred_files = sorted(glob.glob("reports/preds/*_test_preds.csv"))
    if not pred_files:
        raise FileNotFoundError(
            "No prediction files found in reports/preds/. Run compare_models_v1.py first."
        )

    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("reports/figures/regime_slices", exist_ok=True)

    rows = []
    for f in pred_files:
        name = os.path.basename(f).replace("_test_preds.csv", "")
        df = pd.read_csv(f)

        # overall
        rows.append(summarize_slice(df, name, "overall"))

        # slices
        normal = df[df["p_stress"] < 0.5].copy()
        stress = df[df["p_stress"] >= 0.5].copy()

        rows.append(summarize_slice(normal, name, "normal(p_stress<0.5)"))
        rows.append(summarize_slice(stress, name, "stress(p_stress>=0.5)"))

        # calibration plots by slice
        plot_calibration(
            normal,
            title=f"{name} — Calibration (Test, Normal months)",
            outpath=f"reports/figures/regime_slices/{name}_calib_normal.png",
        )
        plot_calibration(
            stress,
            title=f"{name} — Calibration (Test, Stress months)",
            outpath=f"reports/figures/regime_slices/{name}_calib_stress.png",
        )

    out = pd.DataFrame(rows)
    out.to_csv("reports/metrics/test_metrics_by_regime_slice.csv", index=False)

    # Print a compact view: test slices only
    print(out.sort_values(["model", "slice"]))

    print("Saved:")
    print(" - reports/metrics/test_metrics_by_regime_slice.csv")
    print(" - reports/figures/regime_slices/*_calib_normal.png")
    print(" - reports/figures/regime_slices/*_calib_stress.png")


if __name__ == "__main__":
    main()
