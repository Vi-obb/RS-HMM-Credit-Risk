from __future__ import annotations

import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


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
    obs_cols = [column for column in ["inflation", "policy_rate", "unemployment"] if column in macro.columns]
    if len(obs_cols) < 2:
        raise ValueError("Macro HMM requires at least two observed macro columns.")
    x = macro[obs_cols].to_numpy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

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
    return add_stress_dynamics(out), model, scaler
