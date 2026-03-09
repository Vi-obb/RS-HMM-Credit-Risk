from __future__ import annotations

from rs_hmm.config import load_config
from rs_hmm.hmm import fit_macro_hmm
from rs_hmm.labels import make_labels
from rs_hmm.models import prepare_model_frame, train_all_models
from rs_hmm.simulation import run_simulation_from_config


def test_logistic_training_smoke() -> None:
    config = load_config("configs/sim_v1.yml")
    macro, loans, panel = run_simulation_from_config(config)
    labeled = make_labels(panel, config.label.horizon_months, config.behavior.dpd_default)
    macro_hmm, _, _ = fit_macro_hmm(macro)
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
    assert not predictions.empty
    assert not coefficients.empty
