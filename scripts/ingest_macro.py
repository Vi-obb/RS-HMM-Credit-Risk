from __future__ import annotations

import argparse

from rs_hmm.workflow import run_macro_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and normalize FRED macro data.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()
    output = run_macro_ingestion(args.config)
    print(output)


if __name__ == "__main__":
    main()
