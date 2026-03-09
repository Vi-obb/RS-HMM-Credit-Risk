from __future__ import annotations

from pathlib import Path

from rs_hmm.config import load_config


def test_load_config_has_required_sections() -> None:
    config = load_config(Path("configs/sim_v1.yml"))
    assert config.simulation.n_months > 0
    assert config.regimes.n_regimes == 2
    assert config.paths.interim_dir.name == "interim"
