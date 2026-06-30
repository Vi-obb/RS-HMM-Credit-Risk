from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


RAW_FALLBACK_COLUMNS = ["inflation", "policy_rate", "unemployment"]
EXPANDED_TRIGGER_COLUMNS = {
    "house_price_growth",
    "mortgage_rate",
    "mortgage_spread",
    "long_rate",
    "long_rate_change",
    "yield_curve",
    "initial_claims",
    "initial_claims_change",
    "payroll_growth",
    "real_income_growth",
    "financial_conditions",
    "consumer_sentiment",
    "consumer_sentiment_change",
}
STRESS_ORIENTATION = {
    "inflation": 1.0,
    "policy_rate": 1.0,
    "unemployment": 1.0,
    "unemployment_change": 1.0,
    "house_price_growth": -1.0,
    "mortgage_rate": 1.0,
    "mortgage_spread": 1.0,
    "long_rate": 1.0,
    "long_rate_change": 1.0,
    "yield_curve": -1.0,
    "initial_claims": 1.0,
    "initial_claims_change": 1.0,
    "payroll_growth": -1.0,
    "real_income_growth": -1.0,
    "financial_conditions": 1.0,
    "consumer_sentiment": -1.0,
    "consumer_sentiment_change": -1.0,
}
FACTOR_GROUPS = {
    "housing_collateral_factor": ["house_price_growth"],
    "rate_refinancing_factor": [
        "policy_rate",
        "mortgage_rate",
        "mortgage_spread",
        "long_rate",
        "long_rate_change",
        "yield_curve",
    ],
    "labor_income_factor": [
        "unemployment",
        "unemployment_change",
        "initial_claims",
        "initial_claims_change",
        "payroll_growth",
        "real_income_growth",
        "consumer_sentiment",
        "consumer_sentiment_change",
    ],
    "financial_conditions_factor": ["financial_conditions"],
}


@dataclass(frozen=True)
class HMMInputSpec:
    mode: str
    columns: list[str]
    source_columns: list[str]


def _filtered_probabilities(model: GaussianHMM, x_scaled: np.ndarray) -> np.ndarray:
    log_likelihood = model._compute_log_likelihood(x_scaled)
    log_startprob = np.log(np.maximum(model.startprob_, np.finfo(float).tiny))
    log_transmat = np.log(np.maximum(model.transmat_, np.finfo(float).tiny))

    log_alpha = np.empty_like(log_likelihood)
    log_alpha[0] = log_startprob + log_likelihood[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    for t in range(1, len(x_scaled)):
        log_alpha[t] = log_likelihood[t] + logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0)
        log_alpha[t] -= logsumexp(log_alpha[t])

    return np.exp(log_alpha)


def _oriented_z_scores(macro: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    z_scores = pd.DataFrame(index=macro.index)
    for column in columns:
        signed = pd.to_numeric(macro[column], errors="coerce") * STRESS_ORIENTATION[column]
        mean = signed.mean()
        std = signed.std(ddof=0)
        if pd.isna(std) or std == 0.0:
            z_scores[column] = signed.where(signed.isna(), 0.0)
        else:
            z_scores[column] = (signed - mean) / std
    return z_scores


def _add_interpretable_factors(macro: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = macro.copy()
    factor_columns: list[str] = []
    source_columns: list[str] = []
    for factor_column, candidates in FACTOR_GROUPS.items():
        available = [column for column in candidates if column in out.columns]
        if not available:
            continue

        z_scores = _oriented_z_scores(out, available)
        out[factor_column] = z_scores[available].mean(axis=1, skipna=True)
        factor_columns.append(factor_column)
        source_columns.extend(available)

    return out, factor_columns, list(dict.fromkeys(source_columns))


def _select_hmm_inputs(macro: pd.DataFrame) -> tuple[pd.DataFrame, HMMInputSpec]:
    expanded_available = any(column in macro.columns for column in EXPANDED_TRIGGER_COLUMNS)
    if expanded_available:
        with_factors, factor_columns, source_columns = _add_interpretable_factors(macro)
        if len(factor_columns) >= 2:
            return with_factors, HMMInputSpec(
                mode="interpretable_factors",
                columns=factor_columns,
                source_columns=source_columns,
            )
        macro = with_factors

    obs_cols = [column for column in RAW_FALLBACK_COLUMNS if column in macro.columns]
    if len(obs_cols) < 2:
        raise ValueError("Macro HMM requires at least two observed macro columns.")
    return macro, HMMInputSpec(mode="raw", columns=obs_cols, source_columns=obs_cols)


def add_stress_dynamics(macro: pd.DataFrame) -> pd.DataFrame:
    out = macro.copy()
    if "month_date" in out.columns:
        out["month_date"] = pd.to_datetime(out["month_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        out = out.sort_values("month_date").reset_index(drop=True)
    elif "month" in out.columns:
        out = out.sort_values("month").reset_index(drop=True)

    for lag in (3, 6, 12):
        out[f"p_stress_lag_{lag}"] = out["p_stress"].shift(lag)
    out["delta_p_stress"] = out["p_stress"] - out["p_stress"].shift(1)

    stress_run = []
    duration = 0
    for is_stress in out["hard_stress"].fillna(0).astype(int):
        duration = duration + 1 if is_stress else 0
        stress_run.append(duration)
    out["stress_duration"] = stress_run
    return out


def fit_macro_hmm(macro: pd.DataFrame) -> tuple[pd.DataFrame, GaussianHMM, StandardScaler]:
    macro, input_spec = _select_hmm_inputs(macro)
    x_frame = macro[input_spec.columns].apply(pd.to_numeric, errors="coerce")
    x_frame = x_frame.replace([np.inf, -np.inf], np.nan)
    complete = x_frame.notna().all(axis=1)
    if complete.sum() < 2:
        raise ValueError("Macro HMM requires at least two complete macro observations.")
    macro = macro.loc[complete].reset_index(drop=True)
    x_frame = x_frame.loc[complete].reset_index(drop=True)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_frame)

    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=500,
        random_state=42,
    )
    model.fit(x_scaled)

    posterior = _filtered_probabilities(model, x_scaled)
    stress_scores = model.means_.sum(axis=1)
    stress_state = int(np.argmax(stress_scores))

    out = macro.copy()
    out["hmm_state"] = np.argmax(posterior, axis=1)
    out["p_stress"] = posterior[:, stress_state]
    out["hard_stress"] = (out["p_stress"] >= 0.5).astype(int)
    out["hmm_stress_state"] = stress_state
    out["hmm_input_mode"] = input_spec.mode
    out["hmm_input_columns"] = ",".join(input_spec.columns)
    out["hmm_source_columns"] = ",".join(input_spec.source_columns)
    return add_stress_dynamics(out), model, scaler
