from __future__ import annotations

import argparse

from rs_hmm.workflow import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic panel simulation.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()
    outputs = run_simulation(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
