import json
from pathlib import Path

from pii_toolbox.core import DatasetSplit, PiiType
from pii_toolbox.evaluation import tables

TEMPLATE = Path(__file__).parent / "fixtures" / "results" / "spacy_lg__mapa__test__coref-off.json"
SCORES = {"en": 0.5, "fr": 0.3, "de": 0.8}
OFFSETS = {"multi_a": 0.0, "multi_b": 0.01}


def _write_language_file(directory: Path, tool: str, language: str, f2: float) -> None:
    content = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    content["tool"] = tool
    content["detector_languages"] = None
    content["detector_detects"] = [PiiType.PERSON.name]
    for report in ("exact_boundaries", "any_overlap"):
        content[report]["by_language"] = {language: {"precision": f2, "recall": f2 / 2, "f2": f2}}
    (directory / f"{tool}__mapa__{language}__test__coref-off.json").write_text(json.dumps(content), encoding="utf-8")


def _table1(tmp_path: Path):
    for tool in ("multi_a", "multi_b"):
        for language, f2 in SCORES.items():
            _write_language_file(tmp_path, tool, language, f2 + OFFSETS[tool])
    return tables.table1(tables.load_records(tmp_path, DatasetSplit.TEST))


def test_mapa_columns_are_labelled_english(tmp_path):
    header = _table1(tmp_path).header
    assert "MAPA (en) F2 strict" in header
    assert not [column for column in header if column.startswith("MAPA ") and "(en)" not in column]


def test_mapa_columns_equal_each_tools_english_scores(tmp_path):
    table = _table1(tmp_path)
    for row in table.rows:
        cells = dict(zip(table.header, row))
        expected = SCORES["en"] + OFFSETS[row[0]]
        for rule in ("strict", "relaxed"):
            assert cells[f"MAPA (en) P {rule}"] == f"{expected:.3f}"
            assert cells[f"MAPA (en) R {rule}"] == f"{expected / 2:.3f}"
            assert cells[f"MAPA (en) F2 {rule}"] == f"{expected:.3f}"


def test_no_french_or_german_score_appears(tmp_path):
    cells = {cell for row in _table1(tmp_path).rows for cell in row[1:]}
    other_languages = {
        f"{value:.3f}"
        for language in ("fr", "de")
        for offset in OFFSETS.values()
        for value in (SCORES[language] + offset, (SCORES[language] + offset) / 2)
    }
    assert not cells & other_languages
