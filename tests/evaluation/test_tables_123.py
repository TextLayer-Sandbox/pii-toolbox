from pathlib import Path

import pytest

from pii_toolbox.core import DatasetSplit
from pii_toolbox.evaluation import tables
from pii_toolbox.evaluation.tables import build_table1, build_table2, build_table3

FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = FIXTURES / "results"
GOLDEN = FIXTURES / "golden"

CASES = [
    (build_table1, ["table1.md", "table1.csv"]),
    (build_table2, ["table2_strict.md", "table2_strict.csv", "table2_relaxed.md", "table2_relaxed.csv"]),
    (build_table3, ["table3.md", "table3.csv"]),
]


@pytest.mark.parametrize(("builder", "names"), CASES)
def test_tables_match_the_golden_files_byte_for_byte(tmp_path, builder, names):
    written = builder(RESULTS, tmp_path)
    assert [path.name for path in written] == names
    for name in names:
        assert (tmp_path / name).read_bytes() == (GOLDEN / name).read_bytes()


def test_table1_has_precision_recall_and_f2_strict_and_relaxed_for_both_datasets():
    header = tables.table1(tables.load_records(RESULTS, DatasetSplit.TEST)).header
    for dataset in ("TAB", "MAPA (en)"):
        for rule in ("strict", "relaxed"):
            assert [f"{dataset} {metric} {rule}" for metric in ("P", "R", "F2")] == [
                column for column in header if column.startswith(f"{dataset} ") and column.endswith(rule)
            ]


def test_table2_leaves_identifier_blank_for_mapa_and_marks_unreported_types_not_applicable():
    from pii_toolbox.core import SpanMatchRule

    table = tables.table2(tables.load_runs(RESULTS, DatasetSplit.TEST), SpanMatchRule.EXACT_BOUNDARIES)
    identifier = table.header.index("IDENTIFIER")
    mapa_rows = [row for row in table.rows if row[0] == "MAPA"]
    assert mapa_rows and all(row[identifier] == "" for row in mapa_rows)
    presidio = next(row for row in table.rows if row[:2] == ("TAB", "presidio_structured"))
    assert presidio[table.header.index("PERSON")] == "n/a"


def test_table3_has_the_tab_official_columns_and_any_annotator_f2():
    header = tables.table3(tables.load_runs(RESULTS, DatasetSplit.TEST)).header
    assert header[1:4] == ("ER_di", "ER_qi", "WP")
    assert any(column.startswith("Any-annotator F2") for column in header)


def test_coreference_on_runs_and_other_splits_are_left_out(tmp_path):
    build_table3(RESULTS, tmp_path)
    content = (tmp_path / "table3.md").read_text()
    assert "0.990" not in content
    assert "0.110" not in content


def test_removing_a_column_makes_the_golden_comparison_fail(tmp_path, monkeypatch):
    original = tables.table3

    def without_wp(runs):
        table = original(runs)
        keep = [i for i, column in enumerate(table.header) if column != "WP"]
        return tables.Table(
            table.name,
            table.title,
            tuple(table.header[i] for i in keep),
            tuple(tuple(row[i] for i in keep) for row in table.rows),
        )

    monkeypatch.setattr(tables, "table3", without_wp)
    build_table3(RESULTS, tmp_path)
    assert (tmp_path / "table3.md").read_bytes() != (GOLDEN / "table3.md").read_bytes()
