from pathlib import Path

from pii_toolbox.evaluation import tables

FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = FIXTURES / "results_coref"
GOLDEN = FIXTURES / "golden"


def write_table4(out_dir):
    off = tables.load_records(RESULTS, tables.DatasetSplit.TEST)
    on = tables.load_records(RESULTS, tables.DatasetSplit.TEST, coreference=True)
    return tables.write_tables([tables.table4(off, on)], out_dir)


def test_table4_matches_the_golden_files_byte_for_byte(tmp_path):
    written = write_table4(tmp_path)
    assert [path.name for path in written] == ["table4.md", "table4.csv"]
    for name in ("table4.md", "table4.csv"):
        assert (tmp_path / name).read_bytes() == (GOLDEN / name).read_bytes()


def test_rows_come_from_english_data_only(tmp_path):
    write_table4(tmp_path)
    content = (tmp_path / "table4.md").read_text()
    assert "0.664" not in content  # gliner MAPA fr F2, co-reference off
    assert "0.562" not in content  # gliner MAPA de F2, co-reference off
    assert [line.split(" | ")[0] for line in content.splitlines() if line.startswith("| MAPA")] == ["| MAPA", "| MAPA"]


def test_mapa_rows_use_the_english_figures_and_leave_entity_recall_blank(tmp_path):
    off = tables.load_records(RESULTS, tables.DatasetSplit.TEST)
    on = tables.load_records(RESULTS, tables.DatasetSplit.TEST, coreference=True)
    row = next(r for r in tables.table4(off, on).rows if r[:2] == ("MAPA", "gliner"))
    assert row[2:5] == ("0.728", "0.762", "+0.034")
    assert row[5:] == ("",) * 6


def test_a_detector_without_a_co_reference_on_run_has_no_row(tmp_path):
    write_table4(tmp_path)
    assert "presidio_structured" not in (tmp_path / "table4.md").read_text()


def test_tab_rows_show_entity_recall_off_on_and_change(tmp_path):
    off = tables.load_records(RESULTS, tables.DatasetSplit.TEST)
    on = tables.load_records(RESULTS, tables.DatasetSplit.TEST, coreference=True)
    row = next(r for r in tables.table4(off, on).rows if r[:2] == ("TAB", "spacy_lg"))
    assert row[5:] == ("0.900", "0.950", "+0.050", "0.500", "0.600", "+0.100")
