from __future__ import annotations

import argparse

from rs_hmm.workflow import run_model_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and regime-aware logistic models.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()
    outputs = run_model_training(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
