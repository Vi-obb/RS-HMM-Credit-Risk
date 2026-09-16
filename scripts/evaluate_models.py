from __future__ import annotations

import argparse

from rs_hmm.workflow import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved model predictions.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--panel", help="Optional labeled loan-month CSV override.")
    parser.add_argument("--table-dir", help="Optional isolated table directory.")
    parser.add_argument("--figure-dir", help="Optional isolated figure output directory.")
    args = parser.parse_args()
    outputs = run_evaluation(
        args.config,
        panel_path=args.panel,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
