from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rs_hmm.freddie_mac import run_freddie_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Freddie Mac sample loan-level data.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Project-relative path that contains the 'Freddie Mac Mortgage Data' directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim",
        help="Directory for cleaned origination, performance, and panel outputs.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=list(range(2015, 2026)),
        help="Sample years to ingest.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional per-file row limit for a fast smoke run.",
    )
    args = parser.parse_args()

    result = run_freddie_ingestion(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        years=args.years,
        max_rows=args.max_rows,
    )

    print(result.origination_path)
    print(result.performance_path)
    print(result.panel_path)


if __name__ == "__main__":
    main()
