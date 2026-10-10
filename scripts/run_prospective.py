from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import shutil

from rs_hmm.prospective import build_observable_panel, build_realtime_macro, sha256
from rs_hmm.rolling_freeze import PROSPECTIVE_FOLDS, run_rolling_freeze, moving_block_uncertainty
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run the corrected information-timed credit-risk validation")
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = args.data_dir / "loan_monthly_panel_labeled.csv"
    macro = args.data_dir / "macro_realtime.csv"
    if not panel.exists():
        build_observable_panel(args.source_panel, panel, args.data_dir / "outcome_coverage.csv")
    if not (args.data_dir / "outcome_coverage.csv").exists():
        raise RuntimeError("Label preparation is incomplete; coverage audit is missing")
    if not macro.exists():
        build_realtime_macro(macro, args.data_dir / "alfred")
    import importlib.metadata
    manifest = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "hmmlearn")},
                "inputs": {str(path): sha256(path) for path in (args.source_panel, panel, macro, Path("uv.lock"))},
                "folds": [fold.__dict__ for fold in PROSPECTIVE_FOLDS]}
    if manifest["branch"] != "main":
        raise RuntimeError("This rerun must execute on main")
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy2(args.data_dir / "outcome_coverage.csv", args.output_dir / "outcome_coverage.csv")
    shutil.copy2(args.data_dir / "alfred/manifest.json", args.output_dir / "macro_vintage_manifest.json")
    if args.prepare_only:
        return
    run_rolling_freeze(panel, macro, args.output_dir, folds=PROSPECTIVE_FOLDS, require_mature_labels=True)
    monthly = pd.read_csv(args.output_dir / "rolling_origin_paired_monthly_losses.csv")
    pd.concat([moving_block_uncertainty(monthly, block_months=b) for b in (1, 3, 6, 12)]).to_csv(args.output_dir / "bootstrap_block_sensitivity.csv", index=False)
    manifest["outputs"] = {str(path.relative_to(args.output_dir)): sha256(path) for path in args.output_dir.rglob("*.csv")}
    manifest["completed"] = True
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
