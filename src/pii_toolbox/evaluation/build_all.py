"""Rebuilds Tables 1 to 6 and Charts A to F from saved result files, with a manifest of where every cell came from.

Inputs are the result files of one folder plus a cost folder (by default its `cost` subfolder) and nothing else: no
dataset is read and no model is loaded. Only results measured on the test split are accepted,
so every published number comes from TAB test and MAPA test. Outputs are byte-for-byte
reproducible.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from pii_toolbox.core import DatasetSplit, SpanMatchRule
from pii_toolbox.evaluation import charts, tables
from pii_toolbox.evaluation.results_io import ResultRecord, language_from_file_name, read_record
from pii_toolbox.evaluation.tables import Table

COST_FOLDER_NAME = "cost"
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_FILE_NAME = "manifest.json"
PUBLISHED_SPLIT = DatasetSplit.TEST

Records = Mapping[tuple[str, str, str | None, bool], tuple[ResultRecord, str]]
CellSources = Callable[[Sequence[str], str], list[str]]


class NonTestResultError(ValueError):
    """Raised when a result file was measured on a split other than the test split."""


class MissingCostResultsError(FileNotFoundError):
    """Raised when the cost folder does not exist."""


def _index_results(results_dir: Path) -> dict[tuple[str, str, bool], tuple[ResultRecord, str]]:
    """Reads every run result file, refusing any that is not a test-split result; keyed by (tool, dataset, language, co-reference)."""
    index = {}
    for path in sorted(results_dir.glob(tables.RESULT_FILE_PATTERN)):
        record = read_record(path)
        if record.run.split is not PUBLISHED_SPLIT:
            raise NonTestResultError(
                f"{path.name} was measured on the {record.run.split.name.lower()} split; only test results are published"
            )
        index[(record.run.tool, record.run.dataset, record.language, record.run.coreference)] = (record, path.name)
    return index


def _records(index: Records, coreference: bool) -> dict[tuple[str, str, str | None], ResultRecord]:
    return {
        (tool, dataset, language): record
        for (tool, dataset, language, coref), (record, _) in index.items()
        if coref is coreference
    }


def _file(index: Records, tool: str, dataset: str, coreference: bool = False) -> list[str]:
    """The source files of a tool's dataset results; a dataset measured per language has one file per language."""
    return sorted(
        name for (t, d, _, coref), (_, name) in index.items() if (t, d, coref) == (tool, dataset, coreference)
    )


def _table_sources(
    table: Table, cell_sources: CellSources, key_columns: int
) -> dict[str, list[str]]:
    """Maps "row key | column" to the source files of each non-empty cell."""
    mapping = {}
    for row in table.rows:
        row_key = list(row[:key_columns])
        for column, cell in zip(table.header[key_columns:], row[key_columns:]):
            sources = cell_sources(row_key, column)
            if cell != "" and sources:
                mapping[f"{' / '.join(row_key)} | {column}"] = sources
    return mapping


def cost_source_path(cost_file: Path) -> str:
    """Real location of a cost file: relative to the repo root when under it, absolute otherwise."""
    resolved = cost_file.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return resolved.as_posix()


def _write_tables(
    cost_dir: Path, out_dir: Path, index: Records
) -> tuple[list[Table], dict[str, dict[str, list[str]]]]:
    off, on = _records(index, False), _records(index, True)
    strict, relaxed = SpanMatchRule.EXACT_BOUNDARIES, SpanMatchRule.ANY_OVERLAP
    cost_results = tables.load_cost_results(cost_dir)
    off_runs = tables.english_runs(off)

    def named(table: Table, name: str) -> Table:
        return replace(table, name=name)

    def by_dataset_column(row: Sequence[str], column: str) -> list[str]:
        dataset = column.split(" ")[0].lower()
        if dataset == "tab":
            return _file(index, row[0], dataset)
        return [name for name in _file(index, row[0], dataset) if language_from_file_name(name) in (None, "en")]

    def tab_row(row: Sequence[str], column: str) -> list[str]:
        return _file(index, row[0], "tab")

    def dataset_and_tool_row(row: Sequence[str], column: str) -> list[str]:
        return [name for name in _file(index, row[1], row[0].lower()) if language_from_file_name(name) in (None, "en")]

    def coreference_cell(row: Sequence[str], column: str) -> list[str]:
        dataset, tool = row[0].lower(), row[1]
        if column.endswith(" off"):
            return _file(index, tool, dataset)
        if column.endswith(" on"):
            return _file(index, tool, dataset, True)
        return _file(index, tool, dataset) + _file(index, tool, dataset, True)

    def language_cell(row: Sequence[str], column: str) -> list[str]:
        dataset = "tab" if column == tables.TAB_REFERENCE_COLUMN else "mapa"
        return _file(index, row[0], dataset)

    def cost_cell(row: Sequence[str], column: str) -> list[str]:
        return [cost_source_path(cost_dir / f"{row[0].replace('+', '__')}.json")]

    plan: list[tuple[Table, CellSources, int]] = [
        (named(tables.table1(off), "table1"), by_dataset_column, 1),
        (named(tables.table2(off_runs, strict), "table2_strict"), dataset_and_tool_row, 2),
        (named(tables.table2(off_runs, relaxed), "table2_relaxed"), dataset_and_tool_row, 2),
        (named(tables.table3(off_runs), "table3"), tab_row, 1),
        (named(tables.table4(off, on, strict), "table4_strict"), coreference_cell, 2),
        (named(tables.table4(off, on, relaxed), "table4_relaxed"), coreference_cell, 2),
        (named(tables.table5(off, strict), "table5_strict"), language_cell, 1),
        (named(tables.table5(off, relaxed), "table5_relaxed"), language_cell, 1),
        (named(tables.table6(cost_results), "table6"), cost_cell, 1),
    ]
    built, manifest = [], {}
    for table, cell_sources, key_columns in plan:
        built.append(table)
        manifest[table.name] = _table_sources(table, cell_sources, key_columns)
    tables.write_tables(built, out_dir)
    return built, manifest


def _chart_sources(index: Records, cost_dir: Path) -> dict[str, list[str]]:
    tab_off = sorted(name for (tool, dataset, _, coref), (_, name) in index.items() if dataset == "tab" and not coref)
    tab_on = sorted(name for (tool, dataset, _, coref), (_, name) in index.items() if dataset == "tab" and coref)
    mapa_off = sorted(name for (tool, dataset, _, coref), (_, name) in index.items() if dataset == "mapa" and not coref)
    cost = sorted(cost_source_path(path) for path in cost_dir.glob("*.json"))
    return {
        "chart_a.svg": tab_off,
        "chart_b.svg": tab_off,
        "chart_c.svg": tab_off,
        "chart_d.svg": tab_off + cost,
        "chart_e.svg": mapa_off,
        "chart_f.svg": tab_off + tab_on,
    }


def build_outputs(results_dir: Path, out_dir: Path, cost_dir: Path | None = None) -> list[Path]:
    """Writes every table, chart and the manifest into out_dir; returns the files written.

    cost_dir holds the cost result files and defaults to the `cost` subfolder of results_dir.
    """
    if cost_dir is None:
        cost_dir = results_dir / COST_FOLDER_NAME
    if not cost_dir.is_dir():
        raise MissingCostResultsError(f"{cost_dir} does not exist")
    index = _index_results(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, table_manifest = _write_tables(cost_dir, out_dir, index)
    chart_paths = [
        charts.build_chart_a(results_dir, out_dir),
        charts.build_chart_b(results_dir, out_dir),
        charts.build_chart_c(results_dir, out_dir),
        charts.build_chart_d(results_dir, cost_dir, out_dir),
        charts.build_chart_e(results_dir, out_dir),
        charts.build_chart_f(results_dir, out_dir),
    ]
    manifest = {"tables": table_manifest, "charts": _chart_sources(index, cost_dir)}
    manifest_path = out_dir / MANIFEST_FILE_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sorted([*out_dir.glob("table*"), *chart_paths, manifest_path])
