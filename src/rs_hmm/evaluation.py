from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


PROBABILITY_METHOD_LABELS = {
    "p": "primary",
    "p_raw": "raw",
    "p_platt": "platt",
    "p_intercept": "intercept",
    "p_intercept_only": "intercept_only",
    "p_isotonic": "isotonic",
}


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def probability_method_label(column: str) -> str:
    if column in PROBABILITY_METHOD_LABELS:
        return PROBABILITY_METHOD_LABELS[column]
    if column.startswith("p_"):
        return column.removeprefix("p_")
    return column


def probability_method_columns(predictions: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in predictions.columns:
        if column != "p" and not column.startswith("p_"):
            continue
        if column.startswith("p_stress"):
            continue
        if not pd.api.types.is_numeric_dtype(predictions[column]):
            continue
        observed = predictions[column].dropna()
        if not observed.empty and not observed.between(0.0, 1.0).all():
            continue
        columns.append(column)

    ordered: list[str] = []
    for column in ["p", "p_raw"]:
        if column in columns:
            ordered.append(column)
    ordered.extend(column for column in columns if column not in ordered)
    return ordered


def probability_metric_values(frame: pd.DataFrame, probability_column: str) -> dict[str, float | int]:
    work = frame[["y", probability_column]].dropna()
    y = work["y"].astype(int).to_numpy()
    p = work[probability_column].astype(float).to_numpy()
    return {
        "n": int(len(work)),
        "event_count": int(y.sum()),
        "event_rate": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
        "roc_auc": safe_auc(y, p),
        "pr_auc": safe_pr_auc(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def brier_decomposition_values(
    frame: pd.DataFrame,
    probability_column: str,
    n_bins: int,
) -> dict[str, float]:
    work = frame[["y", probability_column]].dropna().copy()
    if work.empty:
        return {
            "reliability": float("nan"),
            "resolution": float("nan"),
            "uncertainty": float("nan"),
            "grouped_brier": float("nan"),
        }
    work["bin"] = pd.qcut(
        work[probability_column].rank(method="first"),
        q=min(n_bins, len(work)),
        labels=False,
        duplicates="drop",
    )
    event_rate = float(work["y"].mean())
    reliability = 0.0
    resolution = 0.0
    for _, bin_df in work.groupby("bin"):
        weight = len(bin_df) / len(work)
        pred_mean = float(bin_df[probability_column].mean())
        observed_rate = float(bin_df["y"].mean())
        reliability += weight * (pred_mean - observed_rate) ** 2
        resolution += weight * (observed_rate - event_rate) ** 2
    uncertainty = event_rate * (1.0 - event_rate)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "grouped_brier": float(reliability - resolution + uncertainty),
    }


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
                "roc_auc": safe_auc(y, p),
                "pr_auc": safe_pr_auc(y, p),
                "brier": float(brier_score_loss(y, p)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
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
                    "roc_auc": safe_auc(y, p),
                    "pr_auc": safe_pr_auc(y, p),
                    "brier": float(brier_score_loss(y, p)),
                    "log_loss": float(log_loss(y, p, labels=[0, 1])),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "slice"]).reset_index(drop=True)


def assign_stress_quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    bins = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = numeric.dropna()
    if valid.empty:
        return bins
    if n_bins <= 1 or valid.nunique() <= 1 or len(valid) <= 1:
        bins.loc[valid.index] = 0
        return bins

    q = min(n_bins, int(valid.nunique()), len(valid))
    try:
        assigned = pd.qcut(valid, q=q, labels=False, duplicates="drop")
    except ValueError:
        assigned = pd.qcut(valid.rank(method="dense"), q=q, labels=False, duplicates="drop")
    bins.loc[assigned.index] = assigned.astype("Int64")
    return bins


def compute_stress_gradient_metrics(predictions: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if "p_stress" not in predictions.columns:
        return pd.DataFrame()

    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return pd.DataFrame()

    rows = []
    for model, frame in test.groupby("model", sort=True):
        work = frame.dropna(subset=["y", "p", "p_stress"]).copy()
        if work.empty:
            continue
        work["stress_bin"] = assign_stress_quantile_bins(work["p_stress"], n_bins)
        work = work.dropna(subset=["stress_bin"]).copy()
        if work.empty:
            continue
        bin_count = int(work["stress_bin"].nunique())
        for bin_id, bin_df in work.groupby("stress_bin", sort=True):
            metrics = probability_metric_values(bin_df, "p")
            stress_bin = int(bin_id)
            row = {
                "model": model,
                "split": "test",
                "stress_bin": stress_bin,
                "stress_bin_label": f"q{stress_bin + 1}_of_{bin_count}",
                "stress_bin_count": bin_count,
                "p_stress_min": float(bin_df["p_stress"].min()),
                "p_stress_mean": float(bin_df["p_stress"].mean()),
                "p_stress_max": float(bin_df["p_stress"].max()),
                "hard_stress_share": (
                    float(bin_df["hard_stress"].mean()) if "hard_stress" in bin_df.columns else float("nan")
                ),
            }
            row.update(metrics)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["model", "stress_bin"]).reset_index(drop=True)


def compute_time_bucket_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return pd.DataFrame()

    if "month_date" in test.columns:
        years = pd.to_datetime(test["month_date"], errors="coerce").dt.year
        test["time_bucket"] = years.astype("Int64").astype(str)
    else:
        test["time_bucket"] = (test["month"] // 12).astype(int).astype(str)
    rows = []
    for (model, time_bucket), frame in test.groupby(["model", "time_bucket"], sort=True):
        y = frame["y"].to_numpy()
        p = frame["p"].to_numpy()
        rows.append(
            {
                "model": model,
                "time_bucket": str(time_bucket),
                "n": int(len(frame)),
                "event_rate": float(np.mean(y)),
                "pred_mean": float(np.mean(p)),
                "brier": float(brier_score_loss(y, p)),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "time_bucket"]).reset_index(drop=True)


def compute_monthly_default_count_error(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return pd.DataFrame()

    if "month_date" in test.columns:
        month_date = pd.to_datetime(test["month_date"], errors="coerce")
        test["test_month"] = month_date.dt.to_period("M").astype(str)
    else:
        test["test_month"] = test["month"].astype(str)

    rows = []
    for (model, test_month), frame in test.groupby(["model", "test_month"], sort=True):
        work = frame.dropna(subset=["y", "p"])
        if work.empty:
            continue
        observed_defaults = float(work["y"].sum())
        predicted_defaults = float(work["p"].sum())
        count_error = predicted_defaults - observed_defaults
        n = int(len(work))
        row = {
            "model": model,
            "split": "test",
            "test_month": str(test_month),
            "n": n,
            "observed_defaults": observed_defaults,
            "predicted_defaults": predicted_defaults,
            "count_error": count_error,
            "absolute_count_error": abs(count_error),
            "relative_count_error": count_error / observed_defaults if observed_defaults > 0 else float("nan"),
            "absolute_relative_count_error": (
                abs(count_error) / observed_defaults if observed_defaults > 0 else float("nan")
            ),
            "observed_default_rate": observed_defaults / n,
            "predicted_default_rate": predicted_defaults / n,
            "count_error_rate": count_error / n,
            "absolute_count_error_rate": abs(count_error) / n,
        }
        if "p_stress" in work.columns:
            row["p_stress_mean"] = float(work["p_stress"].mean())
        if "hard_stress" in work.columns:
            row["hard_stress_share"] = float(work["hard_stress"].mean())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["model", "test_month"]).reset_index(drop=True)


def calibration_points(frame: pd.DataFrame, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    return calibration_curve(frame["y"], frame["p"], n_bins=n_bins, strategy="quantile")


def compute_calibration_bins(predictions: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows = []
    for (model, split), frame in predictions.groupby(["model", "split"], sort=True):
        if frame.empty:
            continue
        work = frame.copy()
        work["calibration_bin"] = pd.qcut(
            work["p"].rank(method="first"),
            q=min(n_bins, len(work)),
            labels=False,
            duplicates="drop",
        )
        for bin_id, bin_df in work.groupby("calibration_bin", sort=True):
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "bin": int(bin_id),
                    "n": int(len(bin_df)),
                    "mean_pred": float(bin_df["p"].mean()),
                    "observed_rate": float(bin_df["y"].mean()),
                    "event_count": int(bin_df["y"].sum()),
                }
            )
    return pd.DataFrame(rows)


def compute_calibration_method_metrics(predictions: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows = []
    for probability_column in probability_method_columns(predictions):
        for (model, split), frame in predictions.groupby(["model", "split"], sort=True):
            work = frame.dropna(subset=["y", probability_column])
            if work.empty:
                continue
            metrics = probability_metric_values(work, probability_column)
            decomposition = brier_decomposition_values(work, probability_column, n_bins)
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "calibration_method": probability_method_label(probability_column),
                    "probability_column": probability_column,
                    **metrics,
                    **decomposition,
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["split", "model", "calibration_method", "probability_column"])
        .reset_index(drop=True)
    )


def compute_brier_decomposition(predictions: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows = []
    for (model, split), frame in predictions.groupby(["model", "split"], sort=True):
        if frame.empty:
            continue
        work = frame.copy()
        work["bin"] = pd.qcut(
            work["p"].rank(method="first"),
            q=min(n_bins, len(work)),
            labels=False,
            duplicates="drop",
        )
        event_rate = float(work["y"].mean())
        reliability = 0.0
        resolution = 0.0
        for _, bin_df in work.groupby("bin"):
            weight = len(bin_df) / len(work)
            pred_mean = float(bin_df["p"].mean())
            observed_rate = float(bin_df["y"].mean())
            reliability += weight * (pred_mean - observed_rate) ** 2
            resolution += weight * (observed_rate - event_rate) ** 2
        uncertainty = event_rate * (1.0 - event_rate)
        rows.append(
            {
                "model": model,
                "split": split,
                "n": int(len(work)),
                "reliability": float(reliability),
                "resolution": float(resolution),
                "uncertainty": float(uncertainty),
                "grouped_brier": float(reliability - resolution + uncertainty),
            }
        )
    return pd.DataFrame(rows)


def compute_data_coverage(labeled: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    frame = labeled.copy()
    if "cohort_year" not in frame.columns:
        frame["cohort_year"] = "all"
    target_cols = [column for column in frame.columns if column.startswith("y_")]
    target = target_cols[0] if target_cols else None
    modeled = predictions[predictions["split"].isin(["train", "val", "test"])]
    modeled_keys = modeled[["loan_id", "month"]].drop_duplicates() if not modeled.empty else pd.DataFrame()
    rows = []
    for cohort, cohort_df in frame.groupby("cohort_year", dropna=False, sort=True):
        if modeled_keys.empty:
            usable_rows = 0
        else:
            usable_rows = len(
                cohort_df[["loan_id", "month"]]
                .merge(modeled_keys, on=["loan_id", "month"], how="inner")
                .drop_duplicates()
            )
        event_count = int(cohort_df[target].sum()) if target else 0
        rows.append(
            {
                "cohort": cohort,
                "loans": int(cohort_df["loan_id"].nunique()),
                "loan_months": int(len(cohort_df)),
                "at_risk_loan_months": int(len(cohort_df)),
                "usable_modeling_rows": int(usable_rows),
                "event_count": event_count,
                "event_rate": float(cohort_df[target].mean()) if target else float("nan"),
                "first_month": str(cohort_df.get("month_date", cohort_df["month"]).min()),
                "last_month": str(cohort_df.get("month_date", cohort_df["month"]).max()),
            }
        )
    return pd.DataFrame(rows)
