from __future__ import annotations

import numpy as np

from rs_hmm.config import load_config
from rs_hmm.simulation import run_simulation_from_config, simulate_regimes


def test_transition_rows_sum_to_one() -> None:
    config = load_config("configs/sim_v1.yml")
    matrix = np.asarray(config.regimes.transition_matrix)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_run_simulation_from_config_shapes() -> None:
    config = load_config("configs/sim_v1.yml")
    macro, loans, panel = run_simulation_from_config(config)
    assert len(macro) == config.simulation.n_months
    assert len(loans) == config.simulation.n_loans
    assert {"loan_id", "month", "dpd"}.issubset(panel.columns)


def test_simulate_regimes_length() -> None:
    config = load_config("configs/sim_v1.yml")
    z = simulate_regimes(25, np.asarray(config.regimes.transition_matrix), np.random.default_rng(7))
    assert len(z) == 25
