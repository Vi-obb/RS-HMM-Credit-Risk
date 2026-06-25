from __future__ import annotations

from pathlib import Path

import pandas as pd

from rs_hmm.config import AppConfig, ensure_output_dirs, load_config
from rs_hmm.freddie_mac import run_freddie_ingestion as run_freddie_ingestion_files
from rs_hmm.evaluation import (
    compute_brier_decomposition,
    compute_calibration_bins,
    compute_data_coverage,
    compute_metrics_table,
    compute_regime_slice_metrics,
    compute_time_bucket_metrics,
)
from rs_hmm.hmm import fit_macro_hmm
from rs_hmm.labels import make_labels
from rs_hmm.macro import fetch_fred_macro
from rs_hmm.models import prepare_model_frame, train_all_models
from rs_hmm.plots import plot_calibration, plot_default_rate, plot_pr, plot_regime_paths, plot_roc
from rs_hmm.simulation import run_simulation_from_config

STUDY_WINDOW_MARGIN_MONTHS = 12

EMPIRICAL_PANEL_COLUMNS = {
    "loan_sequence_number",
    "reporting_month",
    "current_loan_delinquency_status",
    "is_terminated",
    "cohort_year",
    "credit_score",
    "mi_percent",
    "number_of_units",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "original_loan_term",
    "number_of_borrowers",
    "loan_age_months",
    "remaining_months_to_legal_maturity",
    "current_actual_upb",
    "current_interest_rate",
}


def _config(config_path: str | Path) -> AppConfig:
    config = load_config(config_path)
    ensure_output_dirs(config)
    return config


def run_simulation(config_path: str | Path) -> dict[str, Path]:
    config = _config(config_path)
    macro, loans, panel = run_simulation_from_config(config)

    macro_path = config.paths.interim_dir / "macro_monthly.csv"
    loans_path = config.paths.interim_dir / "loans_static.csv"
    panel_path = config.paths.interim_dir / "loan_monthly_panel.csv"

    macro.to_csv(macro_path, index=False)
    loans.to_csv(loans_path, index=False)
    panel.to_csv(panel_path, index=False)

    plot_default_rate(panel, str(config.paths.figure_dir / "monthly_default_share.png"))

    return {
        "macro": macro_path,
        "loans": loans_path,
        "panel": panel_path,
    }


def run_freddie_ingestion(
    data_root: str | Path,
    output_dir: str | Path,
    years: list[int] | None = None,
    max_rows: int | None = None,
) -> dict[str, Path]:
    result = run_freddie_ingestion_files(
        data_root=data_root,
        output_dir=output_dir,
        years=years,
        max_rows=max_rows,
    )
    return {
        "origination": result.origination_path,
        "performance": result.performance_path,
        "panel": result.panel_path,
    }


def run_build_labels(config_path: str | Path) -> Path:
    config = _config(config_path)
    panel_path = config.paths.interim_dir / "loan_monthly_panel.csv"
    if not panel_path.exists():
        panel_path = config.paths.interim_dir / "freddie_loan_month_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Missing loan-month panel at {panel_path}. Run "
            "`python -m scripts.run_experiment --config configs/empirical_v1.yml --ingest-freddie` "
            "after placing Freddie Mac sample files under data/Freddie Mac Mortgage Data/."
        )
    if panel_path.name.startswith("freddie_"):
        panel = pd.read_csv(panel_path, usecols=lambda column: column in EMPIRICAL_PANEL_COLUMNS)
    else:
        panel = pd.read_csv(panel_path)
    labeled = make_labels(
        panel,
        horizon=config.label.horizon_months,
        dpd_default=config.behavior.dpd_default,
    )
    output_path = config.paths.processed_dir / "loan_monthly_panel_labeled.csv"
    labeled.to_csv(output_path, index=False)
    plot_default_rate(labeled, str(config.paths.figure_dir / "monthly_default_share.png"))
    return output_path


def run_macro_ingestion(config_path: str | Path) -> Path:
    config = _config(config_path)
    return fetch_fred_macro(
        output_path=config.paths.interim_dir / "macro_monthly.csv",
        start=None,
        end=None,
    )


def run_hmm_fit(config_path: str | Path) -> Path:
    config = _config(config_path)
    macro_path = config.paths.interim_dir / "macro_monthly.csv"
    if not macro_path.exists():
        run_macro_ingestion(config_path)
    macro = pd.read_csv(macro_path)
    labeled_path = config.paths.processed_dir / "loan_monthly_panel_labeled.csv"
    if labeled_path.exists() and "month_date" in macro.columns:
        labeled_dates = pd.to_datetime(
            pd.read_csv(labeled_path, usecols=["month_date"])["month_date"],
            errors="coerce",
        ).dropna()
        if not labeled_dates.empty:
            macro_dates = pd.to_datetime(macro["month_date"], errors="coerce")
            start = labeled_dates.min() - pd.DateOffset(months=STUDY_WINDOW_MARGIN_MONTHS)
            end = labeled_dates.max()
            macro = macro.loc[(macro_dates >= start) & (macro_dates <= end)].copy()
    macro_hmm, _, _ = fit_macro_hmm(macro)
    output_path = config.paths.processed_dir / "macro_with_hmm.csv"
    macro_hmm.to_csv(output_path, index=False)
    plot_regime_paths(macro_hmm, str(config.paths.figure_dir / "hmm_stress_probability.png"))
    return output_path


def run_model_training(config_path: str | Path) -> dict[str, Path]:
    config = _config(config_path)
    labeled = pd.read_csv(config.paths.processed_dir / "loan_monthly_panel_labeled.csv")
    loans_path = config.paths.interim_dir / "loans_static.csv"
    loans = pd.read_csv(loans_path) if loans_path.exists() else None
    macro_hmm = pd.read_csv(config.paths.processed_dir / "macro_with_hmm.csv")

    model_frame = prepare_model_frame(
        labeled=labeled,
        loans=loans,
        macro_hmm=macro_hmm,
        regime_threshold=config.evaluation.regime_threshold,
        horizon=config.label.horizon_months,
    )
    predictions, coefficients = train_all_models(
        model_frame,
        target=f"y_{config.label.horizon_months}m",
        train_frac=config.split.train_frac,
        val_frac=config.split.val_frac,
    )

    predictions_path = config.paths.table_dir / "model_predictions.csv"
    coefficients_path = config.paths.table_dir / "model_coefficients.csv"
    selection_path = config.paths.table_dir / "model_selection.csv"
    selection_features = {
        "alpha",
        "C",
        "validation_brier_for_selection",
        "tuning_train_rows",
        "tuning_val_rows",
        "platt_intercept",
        "platt_logit_slope",
    }
    model_selection = coefficients[coefficients["feature"].isin(selection_features)].copy()
    predictions.to_csv(predictions_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    model_selection.to_csv(selection_path, index=False)
    return {"predictions": predictions_path, "coefficients": coefficients_path, "selection": selection_path}


def run_evaluation(config_path: str | Path) -> dict[str, Path]:
    config = _config(config_path)
    predictions = pd.read_csv(config.paths.table_dir / "model_predictions.csv")
    labeled = pd.read_csv(config.paths.processed_dir / "loan_monthly_panel_labeled.csv")

    metrics = compute_metrics_table(predictions)
    regime_metrics = compute_regime_slice_metrics(predictions)
    time_metrics = compute_time_bucket_metrics(predictions)
    calibration_bins = compute_calibration_bins(predictions, config.evaluation.n_calibration_bins)
    brier_decomposition = compute_brier_decomposition(predictions, config.evaluation.n_calibration_bins)
    data_coverage = compute_data_coverage(labeled, predictions)

    metrics_path = config.paths.table_dir / "model_metrics.csv"
    regime_metrics_path = config.paths.table_dir / "metrics_by_regime.csv"
    time_metrics_path = config.paths.table_dir / "metrics_by_time_bucket.csv"
    calibration_bins_path = config.paths.table_dir / "calibration_bins.csv"
    brier_decomposition_path = config.paths.table_dir / "brier_decomposition.csv"
    data_coverage_path = config.paths.table_dir / "data_coverage.csv"

    metrics.to_csv(metrics_path, index=False)
    regime_metrics.to_csv(regime_metrics_path, index=False)
    time_metrics.to_csv(time_metrics_path, index=False)
    calibration_bins.to_csv(calibration_bins_path, index=False)
    brier_decomposition.to_csv(brier_decomposition_path, index=False)
    data_coverage.to_csv(data_coverage_path, index=False)

    plot_calibration(
        predictions,
        str(config.paths.figure_dir / "model_calibration_test.png"),
        n_bins=config.evaluation.n_calibration_bins,
    )
    plot_roc(predictions, str(config.paths.figure_dir / "model_roc_test.png"))
    plot_pr(predictions, str(config.paths.figure_dir / "model_pr_test.png"))

    return {
        "data_coverage": data_coverage_path,
        "metrics": metrics_path,
        "metrics_by_regime": regime_metrics_path,
        "metrics_by_time_bucket": time_metrics_path,
        "calibration_bins": calibration_bins_path,
        "brier_decomposition": brier_decomposition_path,
    }
