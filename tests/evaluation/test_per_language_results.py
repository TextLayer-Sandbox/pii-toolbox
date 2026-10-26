import json
from pathlib import Path

from pii_toolbox.core import DatasetSplit, PiiType
from pii_toolbox.evaluation import charts, tables
from pii_toolbox.evaluation.results_io import read_record

TEMPLATE = Path(__file__).parent / "fixtures" / "results" / "spacy_lg__mapa__test__coref-off.json"


def _write_language_file(directory: Path, tool: str, language: str, f2: float, supported: list[str] | None) -> None:
    content = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    content["tool"] = tool
    content["detector_languages"] = supported
    content["detector_detects"] = [PiiType.PERSON.name]
    for report in ("exact_boundaries", "any_overlap"):
        content[report]["by_language"] = {language: {"precision": f2, "recall": f2, "f2": f2}}
    name = f"{tool}__mapa__{language}__test__coref-off.json"
    (directory / name).write_text(json.dumps(content), encoding="utf-8")


def _results(tmp_path: Path) -> Path:
    for language, f2 in (("en", 0.5), ("fr", 0.25), ("de", 0.75)):
        _write_language_file(tmp_path, "multi", language, f2, None)
    _write_language_file(tmp_path, "english_only", "en", 0.125, ["en"])
    return tmp_path


def _row(table, tool):
    return dict(zip(table.header, next(row for row in table.rows if row[0] == tool)))


def test_loader_keeps_one_record_per_tool_dataset_and_language(tmp_path):
    records = tables.load_records(_results(tmp_path), DatasetSplit.TEST)
    assert sorted(records) == [
        ("english_only", "mapa", "en"),
        ("multi", "mapa", "de"),
        ("multi", "mapa", "en"),
        ("multi", "mapa", "fr"),
    ]
    assert records[("multi", "mapa", "fr")].language == "fr"


def test_a_file_name_without_a_language_keeps_language_none(tmp_path):
    path = tmp_path / "gliner__tab__test__coref-off.json"
    path.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    assert read_record(path).language is None


def test_table5_shows_a_separate_value_per_language_with_en_fr_de_first(tmp_path):
    table = tables.table5(tables.load_records(_results(tmp_path), DatasetSplit.TEST))
    assert table.header == ("Tool", "en", "fr", "de", "TAB (English reference)")
    cells = _row(table, "multi")
    assert (cells["en"], cells["fr"], cells["de"]) == ("0.500", "0.250", "0.750")


def test_table5_keeps_not_applicable_for_an_english_only_tool(tmp_path):
    cells = _row(tables.table5(tables.load_records(_results(tmp_path), DatasetSplit.TEST)), "english_only")
    assert (cells["en"], cells["fr"], cells["de"]) == ("0.125", "n/a", "n/a")


def test_chart_e_draws_every_language_of_a_tool_measured_per_language(tmp_path):
    runs = charts._runs_of(_results(tmp_path), "mapa", DatasetSplit.TEST)
    multi = next(run for run in runs if run.tool == "multi")
    assert sorted(multi.exact_boundaries.by_language) == ["de", "en", "fr"]
    assert multi.any_overlap.by_language["fr"].f2 == 0.25
