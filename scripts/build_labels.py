from __future__ import annotations

import argparse

from rs_hmm.workflow import run_build_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward default labels.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()
    output = run_build_labels(args.config)
    print(output)


if __name__ == "__main__":
    main()
