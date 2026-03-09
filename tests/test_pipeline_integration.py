from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    project_root = Path.cwd()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "sim_test.yml"

    raw_config = yaml.safe_load((project_root / "configs" / "sim_v1.yml").read_text(encoding="utf-8"))
    raw_config["paths"] = {
        "raw_dir": str(tmp_path / "data" / "raw"),
        "interim_dir": str(tmp_path / "data" / "interim"),
        "processed_dir": str(tmp_path / "data" / "processed"),
        "figure_dir": str(tmp_path / "reports" / "figures"),
        "table_dir": str(tmp_path / "reports" / "tables"),
    }
    config_path.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")

    commands = [
        [sys.executable, "-m", "scripts.simulate", "--config", str(config_path)],
        [sys.executable, "-m", "scripts.build_labels", "--config", str(config_path)],
        [sys.executable, "-m", "scripts.fit_hmm", "--config", str(config_path)],
        [sys.executable, "-m", "scripts.train_models", "--config", str(config_path)],
        [sys.executable, "-m", "scripts.evaluate_models", "--config", str(config_path)],
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    env["MPLCONFIGDIR"] = str(tmp_path / ".mpl-cache")
    env["XDG_CACHE_HOME"] = str(tmp_path / ".cache")
    env["MPLBACKEND"] = "Agg"
    for command in commands:
        subprocess.run(command, cwd=project_root, check=True, env=env)

    assert (tmp_path / "data" / "processed" / "loan_monthly_panel_labeled.csv").exists()
    assert (tmp_path / "data" / "processed" / "macro_with_hmm.csv").exists()
    assert (tmp_path / "reports" / "tables" / "model_metrics.csv").exists()
    assert (tmp_path / "reports" / "figures" / "model_calibration_test.png").exists()
    assert (tmp_path / "reports" / "figures" / "hmm_stress_probability.png").exists()
