from __future__ import annotations

import os

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def fit_macro_hmm(macro: pd.DataFrame) -> tuple[pd.DataFrame, GaussianHMM, StandardScaler]:
    obs_cols = ["inflation", "policy_rate"]
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

    posterior = model.predict_proba(x_scaled)
    means_original_scale = scaler.inverse_transform(model.means_)
    stress_state = int(np.argmax(means_original_scale[:, 0]))

    out = macro.copy()
    out["hmm_state"] = np.argmax(posterior, axis=1)
    out["p_stress"] = posterior[:, stress_state]
    out["hard_stress"] = (out["p_stress"] >= 0.5).astype(int)
    out["hmm_stress_state"] = stress_state
    return out, model, scaler
