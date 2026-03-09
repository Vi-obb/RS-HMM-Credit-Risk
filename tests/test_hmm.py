from __future__ import annotations

from rs_hmm.config import load_config
from rs_hmm.hmm import fit_macro_hmm
from rs_hmm.simulation import run_simulation_from_config


def test_hmm_outputs_posterior_columns() -> None:
    config = load_config("configs/sim_v1.yml")
    macro, _, _ = run_simulation_from_config(config)
    fitted, _, _ = fit_macro_hmm(macro)
    assert {"p_stress", "hard_stress", "hmm_state"}.issubset(fitted.columns)
    assert fitted["p_stress"].between(0.0, 1.0).all()
