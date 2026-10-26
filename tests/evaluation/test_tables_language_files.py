import shutil
from pathlib import Path

from pii_toolbox.evaluation.tables import build_table2, build_table3

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


def _rows(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()[4:]
    return [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]


def test_table2_has_a_filled_row_per_tool_from_english_files(tmp_path):
    out = tmp_path / "out"
    build_table2(_english_files(tmp_path), out)
    for variant in ("strict", "relaxed"):
        rows = _rows(out / f"table2_{variant}.md")
        tab_rows = [row for row in rows if row[0] == "TAB"]
        assert sorted(row[1] for row in tab_rows) == sorted(TOOLS)
        assert [row[1] for row in rows if row[0] == "MAPA"] == ["spacy_lg"]
        for row in rows:
            assert any(cell not in ("", "n/a") for cell in row[2:])


def test_table3_has_a_filled_row_per_tool_from_english_files(tmp_path):
    out = tmp_path / "out"
    build_table3(_english_files(tmp_path), out)
    rows = _rows(out / "table3.md")
    assert sorted(row[0] for row in rows) == sorted(TOOLS)
    for row in rows:
        assert all(cell not in ("", "n/a") for cell in row[1:])
