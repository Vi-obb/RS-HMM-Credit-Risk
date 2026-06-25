from __future__ import annotations

import argparse

from rs_hmm.workflow import (
    run_build_labels,
    run_evaluation,
    run_freddie_ingestion,
    run_hmm_fit,
    run_macro_ingestion,
    run_model_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the empirical experiment pipeline.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--skip-macro-ingestion",
        action="store_true",
        help="Use an existing data/interim/macro_monthly.csv instead of fetching FRED.",
    )
    parser.add_argument(
        "--ingest-freddie",
        action="store_true",
        help="Ingest Freddie Mac sample files before building labels.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Path containing the 'Freddie Mac Mortgage Data' directory.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=list(range(2015, 2026)),
        help="Freddie sample years to ingest when --ingest-freddie is used.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional per-file row limit for a smoke run.",
    )
    args = parser.parse_args()

    if args.ingest_freddie:
        for name, path in run_freddie_ingestion(
            data_root=args.data_root,
            output_dir="data/interim",
            years=args.years,
            max_rows=args.max_rows,
        ).items():
            print(f"{name}: {path}")
    if not args.skip_macro_ingestion:
        print(f"macro: {run_macro_ingestion(args.config)}")
    print(f"labels: {run_build_labels(args.config)}")
    print(f"hmm: {run_hmm_fit(args.config)}")
    for name, path in run_model_training(args.config).items():
        print(f"{name}: {path}")
    for name, path in run_evaluation(args.config).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
