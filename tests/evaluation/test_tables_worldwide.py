import json
from pathlib import Path

from pii_toolbox.core import DatasetSplit, SpanMatchRule
from pii_toolbox.evaluation import tables

TEMPLATE = Path(__file__).parent / "fixtures" / "results" / "spacy_lg__mapa__test__coref-off.json"
OFF_F2, ON_F2 = 0.40, 0.55


def _write_worldwide_file(directory: Path, tool: str, coreference: bool, f2: float) -> None:
    content = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    content["tool"] = tool
    content["dataset"] = "worldwide"
    content["coreference"] = coreference
    content["detector_languages"] = None
    for report in ("exact_boundaries", "any_overlap"):
        content[report]["overall_micro"] = {"precision": f2, "recall": f2 / 2, "f2": f2}
        content[report]["by_language"] = {}
    state = "on" if coreference else "off"
    (directory / f"{tool}__worldwide__en__test__coref-{state}.json").write_text(json.dumps(content), encoding="utf-8")


def _records(tmp_path: Path, coreference: bool = False):
    for tool in ("multi_a", "multi_b"):
        _write_worldwide_file(tmp_path, tool, False, OFF_F2)
        _write_worldwide_file(tmp_path, tool, True, ON_F2)
    return tables.load_records(tmp_path, DatasetSplit.TEST, coreference)


def test_table1_has_three_worldwide_strict_and_three_relaxed_columns_after_mapa(tmp_path):
    header = tables.table1(_records(tmp_path)).header
    worldwide = [column for column in header if column.startswith("Worldwide ")]
    assert worldwide == [f"Worldwide {metric} {rule}" for rule in ("strict", "relaxed") for metric in ("P", "R", "F2")]
    assert header.index(worldwide[0]) == max(i for i, column in enumerate(header) if column.startswith("MAPA ")) + 1


def test_table1_worldwide_cells_hold_the_runs_overall_scores(tmp_path):
    table = tables.table1(_records(tmp_path))
    for row in table.rows:
        cells = dict(zip(table.header, row))
        for rule in ("strict", "relaxed"):
            assert cells[f"Worldwide P {rule}"] == f"{OFF_F2:.3f}"
            assert cells[f"Worldwide R {rule}"] == f"{OFF_F2 / 2:.3f}"
            assert cells[f"Worldwide F2 {rule}"] == f"{OFF_F2:.3f}"


def test_table2_has_worldwide_rows_with_blank_identifier_cells(tmp_path):
    runs = tables.english_runs(_records(tmp_path))
    table = tables.table2(runs, SpanMatchRule.EXACT_BOUNDARIES)
    rows = [dict(zip(table.header, row)) for row in table.rows if row[0] == "Worldwide"]
    assert [row["Tool"] for row in rows] == ["multi_a", "multi_b"]
    assert all(row["IDENTIFIER"] == "" for row in rows)


def test_table4_worldwide_row_shows_off_on_and_change(tmp_path):
    table = tables.table4(_records(tmp_path), _records(tmp_path, coreference=True))
    rows = [dict(zip(table.header, row)) for row in table.rows if row[0] == "Worldwide"]
    assert [row["Detector"] for row in rows] == ["multi_a", "multi_b"]
    for row in rows:
        assert row["F2 off"] == f"{OFF_F2:.3f}"
        assert row["F2 on"] == f"{ON_F2:.3f}"
        assert row["F2 change"] == f"{ON_F2 - OFF_F2:+.3f}"
        assert row["ER_di off"] == ""


def test_no_header_or_cell_mentions_region(tmp_path):
    off, on = _records(tmp_path), _records(tmp_path, coreference=True)
    runs = tables.english_runs(off)
    built = [tables.table1(off), tables.table2(runs, SpanMatchRule.ANY_OVERLAP), tables.table4(off, on)]
    for table in built:
        text = " ".join([*table.header, *(cell for row in table.rows for cell in row)]).lower()
        assert "region" not in text
