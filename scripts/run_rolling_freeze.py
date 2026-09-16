from __future__ import annotations

import argparse

from rs_hmm.rolling_freeze import run_rolling_freeze


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the predefined causal rolling-origin freeze pass.")
    parser.add_argument("--panel", required=True, help="Existing labeled Freddie loan-month CSV.")
    parser.add_argument("--macro", required=True, help="Existing transformed monthly macro CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for candidate freeze tables.")
    args = parser.parse_args()
    for name, path in run_rolling_freeze(args.panel, args.macro, args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
