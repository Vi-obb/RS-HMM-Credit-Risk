from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "baseline_logit": "Borrower-only",
    "regime_aware_logit": "Regime-aware",
    "macro_logit": "Macro-augmented",
}


def _finish(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fold_scores(metrics: pd.DataFrame, output: Path) -> None:
    test = metrics[
        (metrics["split"] == "test")
        & (metrics["fold"] != "pooled")
        & metrics["model"].isin(["baseline_logit", "regime_aware_logit"])
    ].copy()
    folds = list(dict.fromkeys(test["fold"]))
    labels = [fold.replace("origin_", "").replace("_", "-") for fold in folds]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    for model, frame in test.groupby("model", sort=True):
        ordered = frame.set_index("fold").loc[folds]
        axes[0].plot(labels, ordered["brier"], marker="o", label=MODEL_LABELS[model])
        axes[1].plot(labels, ordered["log_loss"], marker="o", label=MODEL_LABELS[model])
    axes[0].set_title("Brier score")
    axes[1].set_title("Log loss")
    for ax in axes:
        ax.set_xlabel("Test origin")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score (lower is better)")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.87, 0.5),
        frameon=False,
    )
    fig.suptitle("Rolling-origin proper scores")
    fig.tight_layout(rect=(0, 0, 0.86, 0.94))
    _finish(fig, output)


def plot_fold_calibration(metrics: pd.DataFrame, output: Path) -> None:
    test = metrics[
        (metrics["split"] == "test")
        & (metrics["fold"] != "pooled")
        & metrics["model"].isin(MODEL_LABELS)
    ].copy()
    folds = list(dict.fromkeys(test["fold"]))
    labels = [fold.replace("origin_", "").replace("_", "-") for fold in folds]
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    event_rate = test.groupby("fold", sort=False)["event_rate"].first().reindex(folds)
    ax.plot(labels, event_rate, color="black", linestyle="--", marker="o", label="Event rate")
    for model, frame in test.groupby("model", sort=True):
        ordered = frame.set_index("fold").loc[folds]
        ax.plot(labels, ordered["pred_mean"], marker="o", label=MODEL_LABELS[model])
    ax.set_xlabel("Test origin")
    ax.set_ylabel("Monthly pooled rate")
    ax.set_title("Mean predicted PD and event rate by origin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 0.76, 1))
    _finish(fig, output)


def plot_monthly_differences(paired: pd.DataFrame, output: Path) -> None:
    work = paired.sort_values("month_date").copy()
    work["month_date"] = pd.to_datetime(work["month_date"])
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), sharex=True)
    axes[0].plot(work["month_date"], work["brier_loss_difference"], color="#1f77b4")
    axes[1].plot(work["month_date"], work["log_loss_difference"], color="#d95f02")
    for ax in axes:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Brier difference")
    axes[1].set_ylabel("Log-loss difference")
    axes[1].set_xlabel("Test month")
    axes[0].set_title("Regime-aware minus borrower-only monthly loss")
    fig.tight_layout()
    _finish(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the frozen rolling-origin result tables.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metrics = pd.read_csv(input_dir / "rolling_origin_metrics.csv")
    paired = pd.read_csv(input_dir / "rolling_origin_paired_monthly_losses.csv")
    plot_fold_scores(metrics, output_dir / "rolling_fold_scores.png")
    plot_fold_calibration(metrics, output_dir / "rolling_fold_calibration.png")
    plot_monthly_differences(paired, output_dir / "rolling_monthly_loss_differences.png")


if __name__ == "__main__":
    main()
