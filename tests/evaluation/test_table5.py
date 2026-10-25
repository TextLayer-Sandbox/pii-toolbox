from pathlib import Path

from pii_toolbox.core import DatasetSplit, SpanMatchRule
from pii_toolbox.evaluation import tables
from pii_toolbox.evaluation.results_io import read_record
from pii_toolbox.evaluation.tables import build_table5

FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = FIXTURES / "results"
GOLDEN = FIXTURES / "golden"


def table():
    return tables.table5(tables.load_records(RESULTS, DatasetSplit.TEST))


def row(tool):
    return dict(zip(table().header, next(r for r in table().rows if r[0] == tool)))


def test_table5_matches_the_golden_file_byte_for_byte(tmp_path):
    written = build_table5(RESULTS, tmp_path)
    assert [path.name for path in written] == ["table5.md", "table5.csv"]
    for name in ("table5.md", "table5.csv"):
        assert (tmp_path / name).read_bytes() == (GOLDEN / name).read_bytes()


def test_columns_put_en_fr_de_first_then_other_languages_then_the_tab_reference():
    assert table().header == ("Tool", "en", "fr", "de", "es", "TAB (English reference)")


def test_english_only_detector_is_not_applicable_outside_english_and_never_zero():
    cells = row("spacy_lg")
    assert cells["fr"] == cells["de"] == cells["es"] == "n/a"
    assert cells["en"] == "0.618"


def test_multilingual_detector_shows_f2_values():
    cells = row("spacy_lg+presidio_structured")
    assert (cells["en"], cells["fr"], cells["de"], cells["es"]) == ("0.728", "0.664", "0.562", "0.461")


def test_not_applicable_is_decided_by_the_languages_recorded_in_the_result_file(tmp_path):
    record = read_record(RESULTS / "spacy_lg__mapa__test__coref-off.json")
    assert record.languages == frozenset({"en"})
    assert read_record(RESULTS / "spacy_lg+presidio_structured__mapa__test__coref-off.json").languages is None


def test_tools_without_a_mapa_run_have_blank_language_cells():
    cells = row("presidio_structured")
    assert cells["en"] == cells["fr"] == ""
    assert cells["TAB (English reference)"] == "0.292"
