from __future__ import annotations

import argparse

from rs_hmm.workflow import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved model predictions.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()
    outputs = run_evaluation(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
