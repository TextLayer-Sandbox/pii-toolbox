import shutil
from pathlib import Path

from pii_toolbox.core import DatasetSplit, SpanMatchRule
from pii_toolbox.evaluation import tables

RESULTS = Path(__file__).parent / "fixtures" / "results"
TOOLS = ("presidio_structured", "spacy_lg")


def _english_files(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    for tool, dataset in [("presidio_structured", "tab"), ("spacy_lg", "tab"), ("spacy_lg", "mapa")]:
        shutil.copy(
            RESULTS / f"{tool}__{dataset}__test__coref-off.json",
            results / f"{tool}__{dataset}__en__test__coref-off.json",
        )
    return results


def _english_runs(results: Path):
    return tables.english_runs(tables.load_records(results, DatasetSplit.TEST))


def _rows(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()[4:]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]


def test_table2_has_a_filled_row_per_tool_from_english_files(tmp_path):
    out = tmp_path / "out"
    runs = _english_runs(_english_files(tmp_path))
    tables.write_tables(
        [tables.table2(runs, SpanMatchRule.EXACT_BOUNDARIES), tables.table2(runs, SpanMatchRule.ANY_OVERLAP)], out
    )
    for variant in ("strict", "relaxed"):
        rows = _rows(out / f"table2_{variant}.md")
        tab_rows = [row for row in rows if row[0] == "TAB"]
        assert sorted(row[1] for row in tab_rows) == sorted(TOOLS)
        assert [row[1] for row in rows if row[0] == "MAPA"] == ["spacy_lg"]
        for row in rows:
            assert any(cell not in ("", "n/a") for cell in row[2:])


def test_table3_has_a_filled_row_per_tool_from_english_files(tmp_path):
    out = tmp_path / "out"
    tables.write_tables([tables.table3(_english_runs(_english_files(tmp_path)))], out)
    rows = _rows(out / "table3.md")
    assert sorted(row[0] for row in rows) == sorted(TOOLS)
    for row in rows:
        assert all(cell not in ("", "n/a") for cell in row[1:])
