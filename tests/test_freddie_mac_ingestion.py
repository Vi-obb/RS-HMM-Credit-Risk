from __future__ import annotations

from pathlib import Path

from rs_hmm.freddie_mac import (
    build_loan_month_panel,
    load_origination_file,
    load_performance_file,
    run_freddie_ingestion,
)


def test_load_sample_files_parses_key_fields() -> None:
    data_root = Path("data/Freddie Mac Mortgage Data")
    origination = load_origination_file(data_root / "sample_2015" / "sample_orig_2015.txt", 2015, max_rows=100)
    performance = load_performance_file(data_root / "sample_2015" / "sample_svcg_2015.txt", 2015, max_rows=100)

    assert not origination.empty
    assert not performance.empty
    assert origination.loc[0, "loan_sequence_number"] == "F15Q10000025"
    assert performance.loc[0, "loan_sequence_number"] == "F15Q10000025"
    assert origination.loc[0, "original_upb"] == 417000
    assert performance.loc[0, "current_actual_upb"] == 417000


def test_build_loan_month_panel_contains_joined_columns() -> None:
    data_root = Path("data/Freddie Mac Mortgage Data")
    origination = load_origination_file(data_root / "sample_2025" / "sample_orig_2025.txt", 2025, max_rows=100)
    performance = load_performance_file(data_root / "sample_2025" / "sample_svcg_2025.txt", 2025, max_rows=100)
    panel = build_loan_month_panel(origination, performance)

    assert {"loan_sequence_number", "reporting_month", "original_upb", "current_actual_upb"}.issubset(panel.columns)
    assert panel["is_90_plus_dpd"].isin([0, 1]).all()


def test_run_freddie_ingestion_writes_clean_tables(tmp_path: Path) -> None:
    result = run_freddie_ingestion(
        data_root=Path("data"),
        output_dir=tmp_path,
        years=[2015],
        max_rows=100,
    )

    assert result.origination_path.exists()
    assert result.performance_path.exists()
    assert result.panel_path.exists()
