from __future__ import annotations

from pathlib import Path

import pandas as pd

from rs_hmm.config import AppConfig, ensure_output_dirs, load_config
from rs_hmm.freddie_mac import run_freddie_ingestion as run_freddie_ingestion_files
from rs_hmm.evaluation import (
    compute_metrics_table,
    compute_regime_slice_metrics,
    compute_time_bucket_metrics,
)
from rs_hmm.hmm import fit_macro_hmm
from rs_hmm.labels import make_labels
from rs_hmm.models import prepare_model_frame, train_all_models
from rs_hmm.plots import plot_calibration, plot_default_rate, plot_regime_paths, plot_roc
from rs_hmm.simulation import run_simulation_from_config


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
    panel = pd.read_csv(config.paths.interim_dir / "loan_monthly_panel.csv")
    labeled = make_labels(
        panel,
        horizon=config.label.horizon_months,
        dpd_default=config.behavior.dpd_default,
    )
    output_path = config.paths.processed_dir / "loan_monthly_panel_labeled.csv"
    labeled.to_csv(output_path, index=False)
    return output_path


def run_hmm_fit(config_path: str | Path) -> Path:
    config = _config(config_path)
    macro = pd.read_csv(config.paths.interim_dir / "macro_monthly.csv")
    macro_hmm, _, _ = fit_macro_hmm(macro)
    output_path = config.paths.processed_dir / "macro_with_hmm.csv"
    macro_hmm.to_csv(output_path, index=False)
    plot_regime_paths(macro_hmm, str(config.paths.figure_dir / "hmm_stress_probability.png"))
    return output_path


def run_model_training(config_path: str | Path) -> dict[str, Path]:
    config = _config(config_path)
    labeled = pd.read_csv(config.paths.processed_dir / "loan_monthly_panel_labeled.csv")
    loans = pd.read_csv(config.paths.interim_dir / "loans_static.csv")
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
    predictions.to_csv(predictions_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    return {"predictions": predictions_path, "coefficients": coefficients_path}


def run_evaluation(config_path: str | Path) -> dict[str, Path]:
    config = _config(config_path)
    predictions = pd.read_csv(config.paths.table_dir / "model_predictions.csv")

    metrics = compute_metrics_table(predictions)
    regime_metrics = compute_regime_slice_metrics(predictions)
    time_metrics = compute_time_bucket_metrics(predictions)

    metrics_path = config.paths.table_dir / "model_metrics.csv"
    regime_metrics_path = config.paths.table_dir / "metrics_by_regime.csv"
    time_metrics_path = config.paths.table_dir / "metrics_by_time_bucket.csv"

    metrics.to_csv(metrics_path, index=False)
    regime_metrics.to_csv(regime_metrics_path, index=False)
    time_metrics.to_csv(time_metrics_path, index=False)

    plot_calibration(
        predictions,
        str(config.paths.figure_dir / "model_calibration_test.png"),
        n_bins=config.evaluation.n_calibration_bins,
    )
    plot_roc(predictions, str(config.paths.figure_dir / "model_roc_test.png"))

    return {
        "metrics": metrics_path,
        "metrics_by_regime": regime_metrics_path,
        "metrics_by_time_bucket": time_metrics_path,
    }
