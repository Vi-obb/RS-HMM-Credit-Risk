from __future__ import annotations

import argparse

from rs_hmm.workflow import run_hmm_fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit macro HMM and export posterior probabilities.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()
    output = run_hmm_fit(args.config)
    print(output)


if __name__ == "__main__":
    main()
