from __future__ import annotations

import argparse

from rs_hmm.workflow import run_build_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward default labels.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--panel", help="Optional input loan-month panel CSV.")
    parser.add_argument("--output", help="Optional output path for the labeled panel CSV.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the default-rate figure.")
    args = parser.parse_args()
    output = run_build_labels(
        args.config,
        panel_path=args.panel,
        output_path=args.output,
        plot=not args.no_plot,
    )
    print(output)


if __name__ == "__main__":
    main()
