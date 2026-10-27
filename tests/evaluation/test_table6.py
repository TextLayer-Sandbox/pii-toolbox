from pathlib import Path

from pii_toolbox.evaluation import tables

FIXTURES = Path(__file__).parent / "fixtures"
COST = FIXTURES / "cost"
GOLDEN = FIXTURES / "golden"


def rows_by_tool():
    table = tables.table6(tables.load_cost_results(COST))
    return table, {row[0]: dict(zip(table.header, row)) for row in table.rows}


def test_table6_matches_the_golden_files_byte_for_byte(tmp_path):
    written = tables.write_tables([tables.table6(tables.load_cost_results(COST))], tmp_path)
    assert [path.name for path in written] == ["table6.md", "table6.csv"]
    for name in ("table6.md", "table6.csv"):
        assert (tmp_path / name).read_bytes() == (GOLDEN / name).read_bytes()


def test_columns_cover_memory_load_throughput_latency_and_lambda():
    table, _ = rows_by_tool()
    assert table.header == (
        "Tool",
        "Peak memory (GiB)",
        "Load time (s)",
        "Throughput at 1k chars (chars/s)",
        "Throughput at 10k chars (chars/s)",
        "Throughput at 100k chars (chars/s)",
        "Latency per 1,000 chars (s)",
        "fits Lambda (10 GB)",
    )


def test_just_under_the_limit_fits_and_just_over_does_not():
    _, rows = rows_by_tool()
    assert rows["just_under_model"]["fits Lambda (10 GB)"] == "yes"
    assert rows["just_over_model"]["fits Lambda (10 GB)"] == "no"
    assert rows["small_model"]["fits Lambda (10 GB)"] == "yes"


def test_single_tools_come_before_combinations():
    table, _ = rows_by_tool()
    names = [row[0] for row in table.rows]
    assert names.index("small_model") < names.index("small_model+presidio_structured")
