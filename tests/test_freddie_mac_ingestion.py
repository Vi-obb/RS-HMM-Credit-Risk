from __future__ import annotations

from pathlib import Path

from rs_hmm.freddie_mac import (
    build_loan_month_panel,
    load_origination_file,
    load_performance_file,
    run_freddie_ingestion,
)


def _write_sample_files(root: Path, year: int, loan_id: str) -> Path:
    sample_dir = root / "Freddie Mac Mortgage Data" / f"sample_{year}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    origination = [""] * 32
    origination[0] = "740"
    origination[1] = f"{year}02"
    origination[3] = f"{year + 30}01"
    origination[5] = "0"
    origination[6] = "1"
    origination[8] = "80"
    origination[9] = "32"
    origination[10] = "417000"
    origination[11] = "80"
    origination[12] = "4.25"
    origination[19] = loan_id
    origination[21] = "360"
    origination[22] = "2"

    performance = [""] * 11
    performance[0] = loan_id
    performance[1] = f"{year}02"
    performance[2] = "417000"
    performance[3] = "0"
    performance[4] = "1"
    performance[5] = "359"
    performance[10] = "4.25"

    (sample_dir / f"sample_orig_{year}.txt").write_text("|".join(origination) + "\n", encoding="utf-8")
    (sample_dir / f"sample_svcg_{year}.txt").write_text("|".join(performance) + "\n", encoding="utf-8")
    return root


def test_load_sample_files_parses_key_fields(tmp_path: Path) -> None:
    data_root = _write_sample_files(tmp_path, 2015, "F15Q10000025") / "Freddie Mac Mortgage Data"
    origination = load_origination_file(data_root / "sample_2015" / "sample_orig_2015.txt", 2015, max_rows=100)
    performance = load_performance_file(data_root / "sample_2015" / "sample_svcg_2015.txt", 2015, max_rows=100)

    assert not origination.empty
    assert not performance.empty
    assert origination.loc[0, "loan_sequence_number"] == "F15Q10000025"
    assert performance.loc[0, "loan_sequence_number"] == "F15Q10000025"
    assert origination.loc[0, "original_upb"] == 417000
    assert performance.loc[0, "current_actual_upb"] == 417000


def test_build_loan_month_panel_contains_joined_columns(tmp_path: Path) -> None:
    data_root = _write_sample_files(tmp_path, 2025, "F25Q10000025") / "Freddie Mac Mortgage Data"
    origination = load_origination_file(data_root / "sample_2025" / "sample_orig_2025.txt", 2025, max_rows=100)
    performance = load_performance_file(data_root / "sample_2025" / "sample_svcg_2025.txt", 2025, max_rows=100)
    panel = build_loan_month_panel(origination, performance)

    assert {"loan_sequence_number", "reporting_month", "original_upb", "current_actual_upb"}.issubset(panel.columns)
    assert panel["is_90_plus_dpd"].isin([0, 1]).all()


def test_run_freddie_ingestion_writes_clean_tables(tmp_path: Path) -> None:
    data_root = _write_sample_files(tmp_path / "raw", 2015, "F15Q10000025")
    result = run_freddie_ingestion(
        data_root=data_root,
        output_dir=tmp_path,
        years=[2015],
        max_rows=100,
    )

    assert result.origination_path.exists()
    assert result.performance_path.exists()
    assert result.panel_path.exists()
