from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import xlogy
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from rs_hmm.empirical import FREDDIE_NUMERIC_FEATURES
from rs_hmm.freddie_mac import FREDDIE_UNAVAILABLE_SENTINELS
from rs_hmm.hmm import FACTOR_GROUPS, STRESS_ORIENTATION, _filtered_probabilities
from rs_hmm.macro import TRANSFORMED_MACRO_COLUMNS
from rs_hmm.models import EMPIRICAL_INTERACTION_BASE


HMM_SEEDS = (11, 23, 42, 71, 101)
ALPHA_CANDIDATES = (3e-6, 1e-5, 3e-5)
TUNING_SAMPLE_ROWS = 500_000
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BLOCK_MONTHS = 3


@dataclass(frozen=True)
class RollingFold:
    name: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str

    def timestamps(self) -> tuple[pd.Timestamp, ...]:
        return tuple(
            pd.Timestamp(value)
            for value in (
                self.train_end,
                self.validation_start,
                self.validation_end,
                self.test_start,
                self.test_end,
            )
        )


ROLLING_FOLDS = (
    RollingFold("origin_2021_04", "2020-03-01", "2020-04-01", "2021-03-01", "2021-04-01", "2022-03-01"),
    RollingFold("origin_2022_04", "2021-03-01", "2021-04-01", "2022-03-01", "2022-04-01", "2023-03-01"),
    RollingFold("origin_2023_04", "2022-03-01", "2022-04-01", "2023-03-01", "2023-04-01", "2024-03-01"),
)


@dataclass
class CausalHMMResult:
    path: pd.DataFrame
    seed_scores: pd.DataFrame
    selected_seed: int
    train_log_likelihood: float


def _month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _fit_factor_statistics(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    statistics: dict[str, tuple[float, float]] = {}
    source_columns = sorted({column for columns in FACTOR_GROUPS.values() for column in columns})
    for column in source_columns:
        if column not in train.columns:
            continue
        signed = pd.to_numeric(train[column], errors="coerce") * STRESS_ORIENTATION[column]
        mean = float(signed.mean())
        std = float(signed.std(ddof=0))
        statistics[column] = (mean, std if np.isfinite(std) and std > 0 else 1.0)
    return statistics


def _transform_factors(
    macro: pd.DataFrame,
    statistics: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    factors = pd.DataFrame(index=macro.index)
    for factor, candidates in FACTOR_GROUPS.items():
        available = [column for column in candidates if column in statistics and column in macro.columns]
        if not available:
            continue
        oriented = []
        for column in available:
            mean, std = statistics[column]
            signed = pd.to_numeric(macro[column], errors="coerce") * STRESS_ORIENTATION[column]
            oriented.append((signed - mean) / std)
        factors[factor] = pd.concat(oriented, axis=1).mean(axis=1, skipna=True)
    if len(factors.columns) != len(FACTOR_GROUPS):
        raise ValueError(f"Expected all four interpretable factors; found {list(factors.columns)}")
    return factors


def fit_causal_hmm_for_fold(
    macro: pd.DataFrame,
    fold: RollingFold,
    seeds: tuple[int, ...] = HMM_SEEDS,
) -> CausalHMMResult:
    work = macro.copy()
    work["month_date"] = _month_start(work["month_date"])
    work = work.sort_values("month_date").reset_index(drop=True)
    train_end = pd.Timestamp(fold.train_end)
    test_end = pd.Timestamp(fold.test_end)
    scoring = work[work["month_date"] <= test_end].copy()
    train_mask = scoring["month_date"] <= train_end
    if int(train_mask.sum()) < 36:
        raise ValueError(f"{fold.name} has fewer than 36 macro training months")

    factor_statistics = _fit_factor_statistics(scoring.loc[train_mask])
    factors = _transform_factors(scoring, factor_statistics)
    complete = factors.notna().all(axis=1)
    scoring = scoring.loc[complete].reset_index(drop=True)
    factors = factors.loc[complete].reset_index(drop=True)
    train_mask = scoring["month_date"] <= train_end

    scaler = StandardScaler()
    x_train = scaler.fit_transform(factors.loc[train_mask])
    x_all = scaler.transform(factors)

    candidates: list[tuple[float, int, GaussianHMM]] = []
    rows: list[dict[str, object]] = []
    for seed in seeds:
        model = GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=500,
            random_state=seed,
        )
        error = ""
        try:
            model.fit(x_train)
            score = float(model.score(x_train))
            converged = bool(model.monitor_.converged)
        except (ValueError, FloatingPointError) as exc:
            score = float("nan")
            converged = False
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "fold": fold.name,
                "seed": seed,
                "train_log_likelihood": score,
                "converged": converged,
                "iterations": int(model.monitor_.iter),
                "error": error,
            }
        )
        if converged and np.isfinite(score):
            candidates.append((score, seed, model))
    if not candidates:
        raise RuntimeError(f"No converged deterministic HMM initialisation for {fold.name}")

    score, seed, model = max(candidates, key=lambda item: (item[0], -item[1]))
    posterior = _filtered_probabilities(model, x_all)
    stress_state = int(np.argmax(model.means_.sum(axis=1)))
    path = scoring[["month_date"]].copy()
    path["fold"] = fold.name
    path["p_stress"] = posterior[:, stress_state]
    path["hard_stress"] = (path["p_stress"] >= 0.5).astype("int8")
    path["hmm_state"] = np.argmax(posterior, axis=1).astype("int8")
    path["hmm_stress_state"] = stress_state
    path["selected_seed"] = seed
    path["train_end"] = train_end
    return CausalHMMResult(
        path=path,
        seed_scores=pd.DataFrame(rows),
        selected_seed=seed,
        train_log_likelihood=score,
    )


def _core_features() -> dict[str, tuple[list[str], list[str]]]:
    base = list(FREDDIE_NUMERIC_FEATURES)
    macro = list(TRANSFORMED_MACRO_COLUMNS)
    interactions = list(EMPIRICAL_INTERACTION_BASE) + [
        "current_delinquency_months",
        "recent_60_dpd_3m",
        "months_since_30_dpd",
    ]
    return {
        "baseline_logit": (base, []),
        "macro_logit": (base + macro, []),
        "regime_aware_logit": (base + ["p_stress"], interactions),
    }


def load_freeze_panel(panel_path: str | Path, macro_path: str | Path) -> pd.DataFrame:
    feature_map = _core_features()
    required = {"month_date", "y_12m"}
    for features, interactions in feature_map.values():
        required.update(
            column
            for column in features
            if column not in TRANSFORMED_MACRO_COLUMNS and column != "p_stress"
        )
        required.update(interactions)
    panel_header = pd.read_csv(panel_path, nrows=0).columns
    missing = required.difference(panel_header)
    if missing:
        raise ValueError(f"Labeled panel is missing required columns: {sorted(missing)}")

    dtype = {column: "float32" for column in required if column not in {"month_date", "y_12m"}}
    dtype["y_12m"] = "int8"
    frame = pd.read_csv(panel_path, usecols=sorted(required), dtype=dtype)
    frame["month_date"] = _month_start(frame["month_date"])

    macro = pd.read_csv(macro_path)
    macro["month_date"] = _month_start(macro["month_date"])
    macro_lookup = macro.set_index("month_date")
    for column in TRANSFORMED_MACRO_COLUMNS:
        frame[column] = frame["month_date"].map(macro_lookup[column]).astype("float32")

    frame = frame.dropna(
        subset=["month_date", "y_12m", *TRANSFORMED_MACRO_COLUMNS]
    ).reset_index(drop=True)
    return frame


def _feature_matrix(
    frame: pd.DataFrame,
    model: str,
    imputer: SimpleImputer,
) -> tuple[np.ndarray, list[str]]:
    features, interactions = _core_features()[model]
    matrix = imputer.transform(frame[features]).astype(np.float32, copy=False)
    names = list(features)
    if interactions:
        positions = {name: index for index, name in enumerate(features)}
        p_stress = matrix[:, positions["p_stress"]]
        extra = np.column_stack(
            [p_stress * matrix[:, positions[column]] for column in interactions]
        ).astype(np.float32, copy=False)
        matrix = np.column_stack([matrix, extra]).astype(np.float32, copy=False)
        names.extend([f"p_stress_x_{column}" for column in interactions])
    return matrix, names


def _clip(probability: np.ndarray) -> np.ndarray:
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def _fit_platt(y: np.ndarray, p_raw: np.ndarray) -> LogisticRegression:
    logits = np.log(_clip(p_raw) / (1.0 - _clip(p_raw))).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
    calibrator.fit(logits, y)
    return calibrator


def _apply_platt(calibrator: LogisticRegression, p_raw: np.ndarray) -> np.ndarray:
    logits = np.log(_clip(p_raw) / (1.0 - _clip(p_raw))).reshape(-1, 1)
    return _clip(calibrator.predict_proba(logits)[:, 1])


def _fit_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_name: str,
) -> tuple[SimpleImputer, StandardScaler, SGDClassifier, LogisticRegression, list[str], dict[str, float]]:
    features, _ = _core_features()[model_name]
    y_train = train["y_12m"].to_numpy(dtype=np.int8, copy=False)
    y_validation = validation["y_12m"].to_numpy(dtype=np.int8, copy=False)

    rng = np.random.default_rng(42)
    train_idx = rng.choice(len(train), size=min(TUNING_SAMPLE_ROWS, len(train)), replace=False)
    validation_idx = rng.choice(len(validation), size=min(TUNING_SAMPLE_ROWS, len(validation)), replace=False)
    best_alpha = ALPHA_CANDIDATES[0]
    best_brier = float("inf")
    for alpha in ALPHA_CANDIDATES:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        imputer.fit(train.iloc[train_idx][features])
        x_tune_train, _ = _feature_matrix(train.iloc[train_idx], model_name, imputer)
        x_tune_validation, _ = _feature_matrix(
            validation.iloc[validation_idx], model_name, imputer
        )
        scaler = StandardScaler()
        x_tune_train = scaler.fit_transform(x_tune_train)
        classifier = SGDClassifier(
            loss="log_loss", alpha=alpha, max_iter=30, tol=1e-3, random_state=42
        )
        classifier.fit(x_tune_train, y_train[train_idx])
        p = classifier.predict_proba(scaler.transform(x_tune_validation))[:, 1]
        brier = float(brier_score_loss(y_validation[validation_idx], _clip(p)))
        if brier < best_brier:
            best_alpha = alpha
            best_brier = brier

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(train[features])
    x_train, feature_names = _feature_matrix(train, model_name, imputer)
    x_validation, _ = _feature_matrix(validation, model_name, imputer)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train, copy=False)
    classifier = SGDClassifier(
        loss="log_loss", alpha=best_alpha, max_iter=30, tol=1e-3, random_state=42
    )
    classifier.fit(x_train, y_train)
    del x_train

    p_validation_raw = _clip(classifier.predict_proba(scaler.transform(x_validation, copy=False))[:, 1])
    calibrator = _fit_platt(y_validation, p_validation_raw)
    p_validation = _apply_platt(calibrator, p_validation_raw)
    metadata = {
        "alpha": float(best_alpha),
        "tuning_validation_brier": best_brier,
        "validation_brier": float(brier_score_loss(y_validation, p_validation)),
        "validation_log_loss": float(log_loss(y_validation, p_validation, labels=[0, 1])),
        "platt_intercept": float(calibrator.intercept_[0]),
        "platt_logit_slope": float(calibrator.coef_[0][0]),
    }
    feature_positions = {feature: index for index, feature in enumerate(features)}
    for feature in FREDDIE_UNAVAILABLE_SENTINELS:
        if feature not in feature_positions:
            continue
        metadata[f"imputation_median_{feature}"] = float(
            imputer.statistics_[feature_positions[feature]]
        )
        indicator = f"{feature}_missing"
        if indicator in train.columns:
            metadata[f"training_missing_share_{feature}"] = float(train[indicator].mean())
    return imputer, scaler, classifier, calibrator, feature_names, metadata


def _predict(
    frame: pd.DataFrame,
    model_name: str,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    classifier: SGDClassifier,
    calibrator: LogisticRegression,
) -> np.ndarray:
    matrix, _ = _feature_matrix(frame, model_name, imputer)
    raw = _clip(classifier.predict_proba(scaler.transform(matrix, copy=False))[:, 1])
    return _apply_platt(calibrator, raw)


def _metric_row(model: str, fold: str, split: str, y: np.ndarray, p: np.ndarray) -> dict[str, object]:
    return {
        "model": model,
        "fold": fold,
        "split": split,
        "n": int(len(y)),
        "event_count": int(y.sum()),
        "event_rate": float(y.mean()),
        "pred_mean": float(p.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def _calibration_rows(
    model: str,
    fold: str,
    split: str,
    y: np.ndarray,
    p: np.ndarray,
    bins: int = 10,
) -> list[dict[str, object]]:
    order = np.argsort(p, kind="stable")
    rows: list[dict[str, object]] = []
    for bin_id, indices in enumerate(np.array_split(order, bins)):
        rows.append(
            {
                "model": model,
                "fold": fold,
                "split": split,
                "bin": bin_id,
                "n": int(len(indices)),
                "mean_pred": float(p[indices].mean()),
                "observed_rate": float(y[indices].mean()),
                "event_count": int(y[indices].sum()),
            }
        )
    return rows


def _monthly_rows(
    model: str,
    fold: str,
    dates: pd.Series,
    y: np.ndarray,
    p: np.ndarray,
) -> list[dict[str, object]]:
    work = pd.DataFrame({"month_date": dates.to_numpy(), "y": y, "p": p})
    rows: list[dict[str, object]] = []
    for month, group in work.groupby("month_date", sort=True):
        observed = float(group["y"].sum())
        predicted = float(group["p"].sum())
        rows.append(
            {
                "model": model,
                "fold": fold,
                "month_date": pd.Timestamp(month),
                "n": int(len(group)),
                "event_count": int(observed),
                "event_rate": float(group["y"].mean()),
                "pred_mean": float(group["p"].mean()),
                "brier": float(np.mean((group["p"] - group["y"]) ** 2)),
                "log_loss": float(log_loss(group["y"], group["p"], labels=[0, 1])),
                "predicted_defaults": predicted,
                "count_error": predicted - observed,
                "absolute_count_error": abs(predicted - observed),
            }
        )
    return rows


def _binary_log_loss_values(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    clipped = _clip(p)
    return -(xlogy(y, clipped) + xlogy(1 - y, 1 - clipped))


def _paired_month_rows(
    fold: str,
    dates: pd.Series,
    y: np.ndarray,
    baseline: np.ndarray,
    regime: np.ndarray,
) -> list[dict[str, object]]:
    work = pd.DataFrame(
        {
            "month_date": dates.to_numpy(),
            "brier_diff": (regime - y) ** 2 - (baseline - y) ** 2,
            "log_loss_diff": _binary_log_loss_values(y, regime) - _binary_log_loss_values(y, baseline),
        }
    )
    rows = []
    for month, group in work.groupby("month_date", sort=True):
        rows.append(
            {
                "fold": fold,
                "month_date": pd.Timestamp(month),
                "n": int(len(group)),
                "brier_loss_difference": float(group["brier_diff"].mean()),
                "log_loss_difference": float(group["log_loss_diff"].mean()),
                "brier_loss_difference_sum": float(group["brier_diff"].sum()),
                "log_loss_difference_sum": float(group["log_loss_diff"].sum()),
            }
        )
    return rows


def moving_block_uncertainty(
    monthly: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    block_months: int = BOOTSTRAP_BLOCK_MONTHS,
    seed: int = 20260727,
) -> pd.DataFrame:
    ordered = monthly.sort_values("month_date").reset_index(drop=True)
    n_months = len(ordered)
    if n_months < block_months:
        raise ValueError("Not enough test months for moving-block uncertainty")
    starts = np.arange(n_months - block_months + 1)
    blocks_needed = int(np.ceil(n_months / block_months))
    rng = np.random.default_rng(seed)
    results: list[dict[str, object]] = []
    sum_columns = {
        "brier": "brier_loss_difference_sum",
        "log_loss": "log_loss_difference_sum",
    }
    for metric, sum_column in sum_columns.items():
        observed = float(ordered[sum_column].sum() / ordered["n"].sum())
        samples = np.empty(replicates, dtype=float)
        for rep in range(replicates):
            chosen = rng.choice(starts, size=blocks_needed, replace=True)
            indices = np.concatenate([np.arange(start, start + block_months) for start in chosen])[:n_months]
            sampled = ordered.iloc[indices]
            samples[rep] = sampled[sum_column].sum() / sampled["n"].sum()
        results.append(
            {
                "metric": metric,
                "estimate_regime_minus_baseline": observed,
                "ci_lower_95": float(np.quantile(samples, 0.025)),
                "ci_upper_95": float(np.quantile(samples, 0.975)),
                "bootstrap_probability_regime_better": float(np.mean(samples < 0.0)),
                "months": n_months,
                "block_months": block_months,
                "replicates": replicates,
                "seed": seed,
            }
        )
    return pd.DataFrame(results)


def _stress_diagnostics(path: pd.DataFrame, fold: RollingFold) -> list[dict[str, object]]:
    train_end, val_start, val_end, test_start, test_end = fold.timestamps()
    masks = {
        "train": path["month_date"] <= train_end,
        "validation": path["month_date"].between(val_start, val_end),
        "test": path["month_date"].between(test_start, test_end),
    }
    rows: list[dict[str, object]] = []
    for split, mask in masks.items():
        values = path.loc[mask, "p_stress"].to_numpy(dtype=float)
        entropy = -(xlogy(values, values) + xlogy(1 - values, 1 - values))
        rows.append(
            {
                "fold": fold.name,
                "split": split,
                "months": int(len(values)),
                "p_stress_min": float(values.min()),
                "p_stress_mean": float(values.mean()),
                "p_stress_max": float(values.max()),
                "p_stress_sd": float(values.std(ddof=0)),
                "hard_stress_share": float(np.mean(values >= 0.5)),
                "near_zero_share": float(np.mean(values <= 0.01)),
                "near_one_share": float(np.mean(values >= 0.99)),
                "extreme_share": float(np.mean((values <= 0.001) | (values >= 0.999))),
                "mean_binary_entropy": float(np.mean(entropy)),
            }
        )
    return rows


def run_rolling_freeze(
    panel_path: str | Path,
    macro_path: str | Path,
    output_dir: str | Path,
    folds: tuple[RollingFold, ...] = ROLLING_FOLDS,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    macro = pd.read_csv(macro_path)
    macro["month_date"] = _month_start(macro["month_date"])
    panel = load_freeze_panel(panel_path, macro_path)
    study_start = panel["month_date"].min() - pd.DateOffset(months=12)
    macro = macro[macro["month_date"] >= study_start].reset_index(drop=True)

    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    stress_rows: list[dict[str, object]] = []
    seed_frames: list[pd.DataFrame] = []
    hmm_paths: list[pd.DataFrame] = []
    pooled: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in _core_features()}

    for fold in folds:
        hmm = fit_causal_hmm_for_fold(macro, fold)
        seed_frames.append(hmm.seed_scores)
        hmm_paths.append(hmm.path)
        stress_rows.extend(_stress_diagnostics(hmm.path, fold))

        fold_frame = panel[panel["month_date"] <= pd.Timestamp(fold.test_end)].copy()
        q_lookup = hmm.path.set_index("month_date")["p_stress"]
        fold_frame["p_stress"] = fold_frame["month_date"].map(q_lookup).astype("float32")
        fold_frame = fold_frame.dropna(subset=["p_stress"]).reset_index(drop=True)
        train_end, val_start, val_end, test_start, test_end = fold.timestamps()
        train = fold_frame[fold_frame["month_date"] <= train_end].copy()
        validation = fold_frame[fold_frame["month_date"].between(val_start, val_end)].copy()
        test = fold_frame[fold_frame["month_date"].between(test_start, test_end)].copy()
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError(f"Empty train, validation, or test slice in {fold.name}")

        predictions: dict[str, np.ndarray] = {}
        y_validation = validation["y_12m"].to_numpy(dtype=np.int8, copy=False)
        y_test = test["y_12m"].to_numpy(dtype=np.int8, copy=False)
        for model_name in _core_features():
            imputer, scaler, classifier, calibrator, feature_names, metadata = _fit_model(
                train, validation, model_name
            )
            p_validation = _predict(validation, model_name, imputer, scaler, classifier, calibrator)
            p_test = _predict(test, model_name, imputer, scaler, classifier, calibrator)
            predictions[model_name] = p_test
            metric_rows.append(_metric_row(model_name, fold.name, "validation", y_validation, p_validation))
            metric_rows.append(_metric_row(model_name, fold.name, "test", y_test, p_test))
            calibration_rows.extend(
                _calibration_rows(model_name, fold.name, "validation", y_validation, p_validation)
            )
            calibration_rows.extend(_calibration_rows(model_name, fold.name, "test", y_test, p_test))
            monthly_rows.extend(_monthly_rows(model_name, fold.name, test["month_date"], y_test, p_test))
            pooled[model_name].append((y_test.copy(), p_test.copy()))
            selection_rows.extend(
                {"fold": fold.name, "model": model_name, "quantity": key, "value": value}
                for key, value in metadata.items()
            )
            coefficient_rows.extend(
                {
                    "fold": fold.name,
                    "model": model_name,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                    "basis_kind": (
                        "interaction_product"
                        if feature.startswith("p_stress_x_")
                        else "main"
                    ),
                    "basis_mean": float(mean),
                    "basis_scale": float(scale),
                    "raw_basis_coefficient": float(coefficient / scale),
                    "standardized_model_intercept": float(classifier.intercept_[0]),
                    "raw_basis_intercept": float(
                        classifier.intercept_[0]
                        - np.sum(classifier.coef_[0] * scaler.mean_ / scaler.scale_)
                    ),
                }
                for feature, coefficient, mean, scale in zip(
                    feature_names,
                    classifier.coef_[0],
                    scaler.mean_,
                    scaler.scale_,
                )
            )

        paired_rows.extend(
            _paired_month_rows(
                fold.name,
                test["month_date"],
                y_test,
                predictions["baseline_logit"],
                predictions["regime_aware_logit"],
            )
        )

    for model_name, pieces in pooled.items():
        y = np.concatenate([piece[0] for piece in pieces])
        p = np.concatenate([piece[1] for piece in pieces])
        metric_rows.append(_metric_row(model_name, "pooled", "test", y, p))
        calibration_rows.extend(_calibration_rows(model_name, "pooled", "test", y, p))

    paired = pd.DataFrame(paired_rows)
    uncertainty = moving_block_uncertainty(paired)
    tables = {
        "rolling_origin_metrics": pd.DataFrame(metric_rows),
        "rolling_origin_calibration_bins": pd.DataFrame(calibration_rows),
        "rolling_origin_monthly_metrics": pd.DataFrame(monthly_rows),
        "rolling_origin_paired_monthly_losses": paired,
        "rolling_origin_paired_uncertainty": uncertainty,
        "rolling_origin_model_selection": pd.DataFrame(selection_rows),
        "rolling_origin_coefficients": pd.DataFrame(coefficient_rows),
        "rolling_origin_stress_diagnostics": pd.DataFrame(stress_rows),
        "rolling_origin_hmm_initialisations": pd.concat(seed_frames, ignore_index=True),
        "rolling_origin_hmm_paths": pd.concat(hmm_paths, ignore_index=True),
        "rolling_origin_folds": pd.DataFrame([fold.__dict__ for fold in folds]),
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths
