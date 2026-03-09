from __future__ import annotations

import os
from pathlib import Path

cache_root = Path.cwd() / ".cache"
mpl_cache = Path.cwd() / ".mpl-cache"
cache_root.mkdir(parents=True, exist_ok=True)
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay


def plot_regime_paths(macro_hmm: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(macro_hmm["month"], macro_hmm["p_stress"], label="HMM p(stress)")
    ax.plot(macro_hmm["month"], macro_hmm["regime_true"], linestyle="--", label="True regime")
    ax.set_xlabel("Month")
    ax.set_ylabel("Stress indicator")
    ax.set_title("Stress probability vs true regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_default_rate(panel: pd.DataFrame, output_path: str) -> None:
    summary = panel.groupby("month")["defaulted"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(summary["month"], summary["defaulted"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Observed default share")
    ax.set_title("Monthly default share")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(predictions: pd.DataFrame, output_path: str, n_bins: int) -> None:
    test = predictions[predictions["split"] == "test"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, frame in test.groupby("model", sort=True):
        frac_pos, mean_pred = calibration_curve(
            frame["y"], frame["p"], n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=model)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default frequency")
    ax.set_title("Calibration by model (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(predictions: pd.DataFrame, output_path: str) -> None:
    test = predictions[predictions["split"] == "test"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, frame in test.groupby("model", sort=True):
        RocCurveDisplay.from_predictions(frame["y"], frame["p"], name=model, ax=ax)
    ax.set_title("ROC by model (test)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
