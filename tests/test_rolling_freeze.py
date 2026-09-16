from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from rs_hmm import rolling_freeze
from rs_hmm.rolling_freeze import RollingFold, fit_causal_hmm_for_fold, moving_block_uncertainty


def test_causal_hmm_uses_only_training_data_for_fit() -> None:
    rng = np.random.default_rng(7)
    months = pd.date_range("2015-01-01", periods=84, freq="MS")
    stress = (np.arange(len(months)) >= 50).astype(float)
    macro = pd.DataFrame(
        {
            "month_date": months,
            "house_price_growth": 5 - 4 * stress + rng.normal(0, 0.1, len(months)),
            "policy_rate": 1 + 2 * stress + rng.normal(0, 0.1, len(months)),
            "mortgage_rate": 3 + stress + rng.normal(0, 0.1, len(months)),
            "mortgage_spread": 1 + stress + rng.normal(0, 0.1, len(months)),
            "long_rate": 2 + stress + rng.normal(0, 0.1, len(months)),
            "long_rate_change": rng.normal(0, 0.1, len(months)),
            "yield_curve": 1 - stress + rng.normal(0, 0.1, len(months)),
            "unemployment": 4 + 2 * stress + rng.normal(0, 0.1, len(months)),
            "unemployment_change": rng.normal(0, 0.1, len(months)),
            "initial_claims": 200 + 50 * stress + rng.normal(0, 1, len(months)),
            "initial_claims_change": rng.normal(0, 1, len(months)),
            "payroll_growth": 2 - stress + rng.normal(0, 0.1, len(months)),
            "real_income_growth": 2 - stress + rng.normal(0, 0.1, len(months)),
            "consumer_sentiment": 90 - 20 * stress + rng.normal(0, 1, len(months)),
            "consumer_sentiment_change": rng.normal(0, 1, len(months)),
            "financial_conditions": -0.5 + stress + rng.normal(0, 0.1, len(months)),
        }
    )
    fold = RollingFold(
        "test", "2019-12-01", "2020-01-01", "2020-06-01", "2020-07-01", "2021-12-01"
    )
    first = fit_causal_hmm_for_fold(macro, fold, seeds=(11, 23))
    changed = macro.copy()
    changed.loc[changed["month_date"] > fold.train_end, "financial_conditions"] += 1000
    second = fit_causal_hmm_for_fold(changed, fold, seeds=(11, 23))

    assert first.selected_seed == second.selected_seed
    assert first.train_log_likelihood == second.train_log_likelihood
    assert first.path.loc[first.path["month_date"] <= fold.train_end, "p_stress"].equals(
        second.path.loc[second.path["month_date"] <= fold.train_end, "p_stress"]
    )


def test_moving_block_uncertainty_is_deterministic() -> None:
    monthly = pd.DataFrame(
        {
            "month_date": pd.date_range("2021-01-01", periods=12, freq="MS"),
            "n": [100] * 12,
            "brier_loss_difference_sum": np.linspace(-0.1, 0.05, 12),
            "log_loss_difference_sum": np.linspace(-0.2, 0.1, 12),
        }
    )
    first = moving_block_uncertainty(monthly, replicates=100, seed=9)
    second = moving_block_uncertainty(monthly, replicates=100, seed=9)
    pd.testing.assert_frame_equal(first, second)


def test_rolling_interactions_use_training_imputation(monkeypatch) -> None:
    monkeypatch.setattr(
        rolling_freeze,
        "_core_features",
        lambda: {"regime_aware_logit": (["original_dti", "original_dti_missing", "p_stress"], ["original_dti"])},
    )
    training = pd.DataFrame(
        {
            "original_dti": [20.0, np.nan, 40.0],
            "original_dti_missing": [0, 1, 0],
            "p_stress": [0.1, 0.2, 0.3],
        }
    )
    validation = pd.DataFrame(
        {
            "original_dti": [np.nan],
            "original_dti_missing": [1],
            "p_stress": [0.5],
        }
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(training)

    matrix, names = rolling_freeze._feature_matrix(validation, "regime_aware_logit", imputer)

    assert names == [
        "original_dti",
        "original_dti_missing",
        "p_stress",
        "p_stress_x_original_dti",
    ]
    assert matrix[0].tolist() == [30.0, 1.0, 0.5, 15.0]


def test_rolling_basis_back_transform_preserves_linear_predictor() -> None:
    raw_basis = np.array(
        [
            [20.0, 0.1, 2.0],
            [30.0, 0.5, 15.0],
            [40.0, 0.9, 36.0],
            [25.0, 0.7, 17.5],
        ]
    )
    target = np.array([0, 1, 1, 0])
    scaler = rolling_freeze.StandardScaler().fit(raw_basis)
    classifier = rolling_freeze.SGDClassifier(
        loss="log_loss", alpha=1e-5, max_iter=100, tol=1e-5, random_state=42
    ).fit(scaler.transform(raw_basis), target)

    raw_coefficients = classifier.coef_[0] / scaler.scale_
    raw_intercept = float(
        classifier.intercept_[0]
        - np.sum(classifier.coef_[0] * scaler.mean_ / scaler.scale_)
    )
    standardized_predictor = (
        classifier.intercept_[0] + scaler.transform(raw_basis) @ classifier.coef_[0]
    )
    raw_predictor = raw_intercept + raw_basis @ raw_coefficients

    assert np.allclose(standardized_predictor, raw_predictor)
