from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rs_hmm.empirical import FREDDIE_NUMERIC_FEATURES, align_macro_months
from rs_hmm.macro import TRANSFORMED_MACRO_COLUMNS
from rs_hmm.splits import time_split_by_month


BASE_FEATURES = [
    "dpd",
    "missed_payment",
    "payment_ratio",
    "principal",
    "tenor_months",
    "interest_rate",
    "borrower_risk",
]
MACRO_FEATURES = ["inflation", "policy_rate"]
EMPIRICAL_MACRO_FEATURES = TRANSFORMED_MACRO_COLUMNS
REGIME_SOFT = ["p_stress"]
REGIME_HARD = ["hard_stress"]
STRESS_DYNAMIC_FEATURES = [
    "p_stress_lag_3",
    "p_stress_lag_6",
    "p_stress_lag_12",
    "delta_p_stress",
    "stress_duration",
]
DYNAMIC_STRESS_ROBUSTNESS_FEATURES = ["delta_p_stress", "stress_duration"]
TUNING_SAMPLE_ROWS = 500_000
EMPIRICAL_INTERACTION_BASE = [
    "credit_score",
    "original_cltv",
    "original_dti",
    "original_ltv",
    "original_interest_rate",
]
PAYMENT_HISTORY_INTERACTION_CANDIDATES = [
    "current_delinquency_months",
    "recent_60_dpd_3m",
    "months_since_30_dpd",
    "delinquent_30_months_6m",
    "cumulative_30_dpd_cures",
    "current_delinquency_state",
    "current_delinquency_status",
    "current_dpd",
    "recent_60_dpd_flag",
    "recent_60_dpd",
    "recent_30_dpd_flag",
    "recent_30_dpd",
    "months_since_delinquency",
    "months_since_last_delinquency",
    "rolling_3m_delinquency_rate",
    "rolling_6m_delinquency_rate",
    "rolling_12m_delinquency_rate",
    "cure_flag",
    "recent_cure_flag",
]
PAYMENT_HISTORY_INTERACTION_KEYWORDS = (
    "delinq",
    "dpd",
    "cure",
)
MAX_PAYMENT_HISTORY_INTERACTIONS = 3
PRIMARY_CALIBRATION_METHOD = "platt"
CALIBRATION_METHODS = ("platt", "intercept_only", "isotonic")
CALIBRATION_PROBABILITY_COLUMNS = {
    "platt": "p_platt",
    "intercept_only": "p_intercept_only",
    "isotonic": "p_isotonic",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: list[str]
    interaction_base: list[str] | None = None
    interaction_drivers: list[str] | None = None


@dataclass(frozen=True)
class InterceptOnlyCalibrator:
    intercept: float


MODEL_SPECS = [
    ModelSpec(name="baseline", features=BASE_FEATURES),
    ModelSpec(name="macro", features=BASE_FEATURES + MACRO_FEATURES),
    ModelSpec(
        name="regime_aware",
        features=BASE_FEATURES + MACRO_FEATURES + REGIME_SOFT,
        interaction_base=["dpd", "missed_payment", "payment_ratio", "inflation", "policy_rate"],
    ),
]


def _augment_interactions(
    df: pd.DataFrame,
    base_cols: list[str],
    driver_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    drivers = driver_cols or ["p_stress"]
    interaction_cols = []
    for driver in drivers:
        for column in base_cols:
            new_col = f"{driver}_x_{column}"
            out[new_col] = out[driver] * out[column]
            interaction_cols.append(new_col)
    return out, interaction_cols


def prepare_model_frame(
    labeled: pd.DataFrame,
    loans: pd.DataFrame | None,
    macro_hmm: pd.DataFrame,
    regime_threshold: float,
    horizon: int,
) -> pd.DataFrame:
    df = labeled.copy()
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if loans is not None and {"loan_id", "principal", "tenor_months", "interest_rate", "borrower_risk"}.issubset(loans.columns):
        df = df.merge(
            loans[["loan_id", "principal", "tenor_months", "interest_rate", "borrower_risk"]],
            on="loan_id",
            how="left",
        )

    macro_hmm = align_macro_months(macro_hmm, df)
    macro_cols = ["month", "p_stress", "hard_stress"] + STRESS_DYNAMIC_FEATURES + [
        column for column in EMPIRICAL_MACRO_FEATURES if column in macro_hmm.columns
    ]
    if "month_date" in macro_hmm.columns:
        macro_cols.append("month_date")
    macro_cols = [column for column in dict.fromkeys(macro_cols) if column in macro_hmm.columns]

    merge_keys = ["month_date"] if "month_date" in df.columns and "month_date" in macro_hmm.columns else ["month"]
    df = df.merge(macro_hmm[macro_cols], on=merge_keys, how="left", suffixes=("", "_macro"))
    if "month_macro" in df.columns and "month" not in merge_keys:
        df = df.drop(columns=["month_macro"])

    df["hard_stress"] = (df["p_stress"] >= regime_threshold).astype(int)
    if {"paid_amount", "scheduled_payment"}.issubset(df.columns):
        df["payment_ratio"] = (df["paid_amount"] / (df["scheduled_payment"] + 1e-9)).clip(0.0, 1.0)
    target = f"y_{horizon}m"
    keep_cols = _required_model_columns(df) + ["p_stress", "hard_stress", target]
    return df.dropna(subset=keep_cols).copy()


def _is_empirical_frame(df: pd.DataFrame) -> bool:
    return {"credit_score", "original_upb", "original_interest_rate"}.issubset(df.columns)


def _available(columns: pd.Index, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in columns]


def _payment_history_interaction_features(df: pd.DataFrame, base_features: list[str]) -> list[str]:
    imported_features = set(base_features)
    selected = [
        column
        for column in PAYMENT_HISTORY_INTERACTION_CANDIDATES
        if column in df.columns and column in imported_features
    ]
    if len(selected) < MAX_PAYMENT_HISTORY_INTERACTIONS:
        selected_set = set(selected)
        for column in base_features:
            if column in selected_set:
                continue
            column_lower = column.lower()
            if any(keyword in column_lower for keyword in PAYMENT_HISTORY_INTERACTION_KEYWORDS):
                selected.append(column)
                selected_set.add(column)
            if len(selected) >= MAX_PAYMENT_HISTORY_INTERACTIONS:
                break
    return selected[:MAX_PAYMENT_HISTORY_INTERACTIONS]


def _empirical_model_specs(df: pd.DataFrame) -> list[ModelSpec]:
    base = _available(df.columns, FREDDIE_NUMERIC_FEATURES)
    macro = _available(df.columns, EMPIRICAL_MACRO_FEATURES)
    regime = _available(df.columns, REGIME_SOFT)
    dynamic_stress = _available(df.columns, DYNAMIC_STRESS_ROBUSTNESS_FEATURES)
    interactions = _available(df.columns, EMPIRICAL_INTERACTION_BASE)
    interactions.extend(_payment_history_interaction_features(df, base))
    interactions = list(dict.fromkeys(interactions))
    specs = [
        ModelSpec(name="baseline_logit", features=base),
        ModelSpec(name="macro_logit", features=base + macro),
        ModelSpec(name="baseline_plus_p_stress_logit", features=base + regime),
        ModelSpec(
            name="regime_aware_logit",
            features=base + regime,
            interaction_base=interactions,
        ),
        ModelSpec(
            name="regime_aware_with_raw_macro_logit",
            features=base + macro + regime,
            interaction_base=interactions,
        ),
    ]
    if dynamic_stress:
        specs.append(
            ModelSpec(
                name="regime_aware_dynamic_stress_robustness_logit",
                features=base + regime + dynamic_stress,
                interaction_base=interactions,
            )
        )
    return specs


def _model_specs_for_frame(df: pd.DataFrame) -> list[ModelSpec]:
    if _is_empirical_frame(df):
        return _empirical_model_specs(df)
    return MODEL_SPECS


def _required_model_columns(df: pd.DataFrame) -> list[str]:
    required: list[str] = []
    for spec in _model_specs_for_frame(df):
        required.extend(spec.features)
        if spec.interaction_drivers:
            required.extend(spec.interaction_drivers)
        if spec.interaction_base:
            required.extend(spec.interaction_base)
    return list(dict.fromkeys(required))


def _clip_probabilities(p: np.ndarray) -> np.ndarray:
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = _clip_probabilities(p)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _make_classifier(empirical: bool, alpha: float | None = None, c_value: float | None = None):
    if empirical:
        return SGDClassifier(
            loss="log_loss",
            alpha=alpha if alpha is not None else 1e-5,
            max_iter=30,
            tol=1e-3,
            random_state=42,
        )
    return LogisticRegression(max_iter=2000, solver="lbfgs", C=c_value if c_value is not None else 1.0)


def _make_pipeline(features: list[str], empirical: bool, alpha: float | None = None, c_value: float | None = None) -> Pipeline:
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), features)],
        remainder="drop",
    )
    return Pipeline([("pre", preprocessor), ("model", _make_classifier(empirical, alpha=alpha, c_value=c_value))])


def _tune_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    target: str,
    empirical: bool,
) -> tuple[Pipeline, dict[str, float]]:
    if empirical:
        candidates = [3e-6, 1e-5, 3e-5]
        key = "alpha"
    else:
        candidates = [0.1, 0.3, 1.0, 3.0, 10.0]
        key = "C"

    tune_train = _bounded_sample(train_df)
    tune_val = _bounded_sample(val_df)

    best_pipeline: Pipeline | None = None
    best_score = float("inf")
    best_value = candidates[0]
    for value in candidates:
        kwargs = {"alpha": value} if empirical else {"c_value": value}
        pipeline = _make_pipeline(features, empirical, **kwargs)
        pipeline.fit(tune_train[features], tune_train[target].astype(int))
        val_p = _clip_probabilities(pipeline.predict_proba(tune_val[features])[:, 1])
        score = float(brier_score_loss(tune_val[target].astype(int), val_p))
        if score < best_score:
            best_pipeline = pipeline
            best_score = score
            best_value = value

    if best_pipeline is None:
        raise RuntimeError("Regularization tuning did not produce a fitted model.")
    return best_pipeline, {
        key: float(best_value),
        "validation_brier_for_selection": best_score,
        "tuning_train_rows": float(len(tune_train)),
        "tuning_val_rows": float(len(tune_val)),
    }


def _bounded_sample(df: pd.DataFrame, max_rows: int = TUNING_SAMPLE_ROWS) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42).sort_values(["month", "loan_id"]).reset_index(drop=True)


def _fit_platt_calibrator(y: pd.Series, p: np.ndarray) -> LogisticRegression | None:
    y_int = y.astype(int)
    if y_int.nunique() < 2:
        return None
    logits = _logit(p).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
    calibrator.fit(logits, y_int)
    return calibrator


def _fit_intercept_only_calibrator(y: pd.Series, p: np.ndarray) -> InterceptOnlyCalibrator | None:
    y_int = y.astype(int)
    if y_int.nunique() < 2:
        return None
    logits = _logit(p)
    target = float(y_int.mean())
    low = -30.0
    high = 30.0
    for _ in range(100):
        mid = (low + high) / 2.0
        calibrated_mean = float(_sigmoid(logits + mid).mean())
        if calibrated_mean < target:
            low = mid
        else:
            high = mid
    return InterceptOnlyCalibrator(intercept=float((low + high) / 2.0))


def _fit_isotonic_calibrator(y: pd.Series, p: np.ndarray) -> IsotonicRegression | None:
    y_int = y.astype(int)
    if y_int.nunique() < 2:
        return None
    calibrator = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
    calibrator.fit(_clip_probabilities(p), y_int)
    return calibrator


def _apply_platt_calibrator(calibrator: LogisticRegression | None, p: np.ndarray) -> np.ndarray:
    clipped = _clip_probabilities(p)
    if calibrator is None:
        return clipped
    logits = _logit(clipped).reshape(-1, 1)
    return _clip_probabilities(calibrator.predict_proba(logits)[:, 1])


def _apply_intercept_only_calibrator(calibrator: InterceptOnlyCalibrator | None, p: np.ndarray) -> np.ndarray:
    clipped = _clip_probabilities(p)
    if calibrator is None:
        return clipped
    return _clip_probabilities(_sigmoid(_logit(clipped) + calibrator.intercept))


def _apply_isotonic_calibrator(calibrator: IsotonicRegression | None, p: np.ndarray) -> np.ndarray:
    clipped = _clip_probabilities(p)
    if calibrator is None:
        return clipped
    return _clip_probabilities(calibrator.predict(clipped))


def _fit_calibrators(y: pd.Series, p: np.ndarray) -> dict[str, object | None]:
    return {
        "platt": _fit_platt_calibrator(y, p),
        "intercept_only": _fit_intercept_only_calibrator(y, p),
        "isotonic": _fit_isotonic_calibrator(y, p),
    }


def _apply_calibrators(calibrators: dict[str, object | None], p: np.ndarray) -> dict[str, np.ndarray]:
    platt = calibrators["platt"]
    intercept_only = calibrators["intercept_only"]
    isotonic = calibrators["isotonic"]
    return {
        "platt": _apply_platt_calibrator(platt if isinstance(platt, LogisticRegression) else None, p),
        "intercept_only": _apply_intercept_only_calibrator(
            intercept_only if isinstance(intercept_only, InterceptOnlyCalibrator) else None,
            p,
        ),
        "isotonic": _apply_isotonic_calibrator(isotonic if isinstance(isotonic, IsotonicRegression) else None, p),
    }


def _calibration_metadata(
    y: pd.Series,
    p_raw: np.ndarray,
    calibrated: dict[str, np.ndarray],
    calibrators: dict[str, object | None],
) -> list[dict[str, float]]:
    y_int = y.astype(int).to_numpy()
    rows: list[dict[str, float]] = []
    for method in CALIBRATION_METHODS:
        method_p = calibrated[method]
        rows.append(
            {
                "feature": f"calibration_validation_brier_{method}",
                "coefficient": float(brier_score_loss(y_int, method_p)),
            }
        )
        rows.append(
            {
                "feature": f"calibration_validation_log_loss_{method}",
                "coefficient": float(log_loss(y_int, method_p, labels=[0, 1])),
            }
        )
    rows.append(
        {
            "feature": "calibration_validation_brier_raw",
            "coefficient": float(brier_score_loss(y_int, _clip_probabilities(p_raw))),
        }
    )
    rows.append(
        {
            "feature": "calibration_validation_log_loss_raw",
            "coefficient": float(log_loss(y_int, _clip_probabilities(p_raw), labels=[0, 1])),
        }
    )

    platt = calibrators["platt"]
    if isinstance(platt, LogisticRegression):
        rows.append({"feature": "platt_intercept", "coefficient": float(platt.intercept_[0])})
        rows.append({"feature": "platt_logit_slope", "coefficient": float(platt.coef_[0][0])})
    intercept_only = calibrators["intercept_only"]
    if isinstance(intercept_only, InterceptOnlyCalibrator):
        rows.append({"feature": "intercept_only_shift", "coefficient": intercept_only.intercept})
    isotonic = calibrators["isotonic"]
    if isinstance(isotonic, IsotonicRegression):
        rows.append({"feature": "isotonic_threshold_count", "coefficient": float(len(isotonic.X_thresholds_))})
    return rows


def train_all_models(
    df: pd.DataFrame,
    target: str,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = time_split_by_month(df, train_frac, val_frac)
    predictions: list[pd.DataFrame] = []
    coefficients: list[dict] = []

    empirical = _is_empirical_frame(df)
    for spec in _model_specs_for_frame(df):
        model_df = df
        features = list(spec.features)
        if spec.interaction_base:
            model_df, interaction_cols = _augment_interactions(
                model_df,
                spec.interaction_base,
                spec.interaction_drivers,
            )
            features.extend(interaction_cols)
        train_df, val_df, test_df = time_split_by_month(model_df, train_frac, val_frac)
        split_map = {"train": train_df, "val": val_df, "test": test_df}

        pipeline, tuning = _tune_pipeline(split_map["train"], split_map["val"], features, target, empirical)
        pipeline.fit(split_map["train"][features], split_map["train"][target].astype(int))
        val_raw_p = _clip_probabilities(pipeline.predict_proba(split_map["val"][features])[:, 1])
        calibrators = _fit_calibrators(split_map["val"][target], val_raw_p)
        val_calibrated = _apply_calibrators(calibrators, val_raw_p)

        coef_values = pipeline.named_steps["model"].coef_[0]
        for feature, coef in zip(features, coef_values):
            coefficients.append({"model": spec.name, "feature": feature, "coefficient": float(coef)})
        for name, value in tuning.items():
            coefficients.append({"model": spec.name, "feature": name, "coefficient": value})
        for row in _calibration_metadata(split_map["val"][target], val_raw_p, val_calibrated, calibrators):
            coefficients.append({"model": spec.name, **row})

        for split_name, split_df in split_map.items():
            preds_raw = _clip_probabilities(pipeline.predict_proba(split_df[features])[:, 1])
            calibrated_preds = _apply_calibrators(calibrators, preds_raw)
            prediction_frame = pd.DataFrame(
                {
                    "model": spec.name,
                    "split": split_name,
                    "loan_id": split_df["loan_id"].to_numpy(),
                    "month": split_df["month"].to_numpy(),
                    "y": split_df[target].astype(int).to_numpy(),
                    "p": calibrated_preds[PRIMARY_CALIBRATION_METHOD],
                    "p_raw": preds_raw,
                    "p_stress": split_df["p_stress"].to_numpy(),
                    "hard_stress": split_df["hard_stress"].to_numpy(),
                }
            )
            for method, column in CALIBRATION_PROBABILITY_COLUMNS.items():
                prediction_frame[column] = calibrated_preds[method]
            if "month_date" in split_df.columns:
                prediction_frame["month_date"] = split_df["month_date"].to_numpy()
            if "cohort_year" in split_df.columns:
                prediction_frame["cohort_year"] = split_df["cohort_year"].to_numpy()
            if "regime_true" in split_df.columns:
                prediction_frame["regime_true"] = split_df["regime_true"].to_numpy()
            predictions.append(prediction_frame)

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(coefficients)
