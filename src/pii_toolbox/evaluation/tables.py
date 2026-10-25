"""Tables 1 to 5 of the article, built from saved result files only.

Each builder reads the result JSON files of one split with co-reference off and writes a
Markdown and a CSV file per table. Rows are tools, single detectors first and then
combinations, each in name order; numbers print with three decimals, so the output is
the same on every run.
"""

import csv
import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pii_toolbox.core import DatasetSplit, DetectionReport, DetectionScore, EvaluationRun, PiiType, SpanMatchRule
from pii_toolbox.evaluation.results_io import ResultRecord, read_record

DATASETS = ("tab", "mapa")
DATASET_TITLES = {"tab": "TAB", "mapa": "MAPA"}
NOT_APPLICABLE = "n/a"
ENGLISH = "en"
RESULT_FILE_PATTERN = "*__*__*__coref-*.json"


@dataclass(frozen=True)
class Table:
    """A titled grid of text cells with one header row."""

    name: str
    title: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def to_markdown(self) -> str:
        lines = [f"**{self.title}**", "", _markdown_row(self.header), _markdown_row(["---"] * len(self.header))]
        lines += [_markdown_row(row) for row in self.rows]
        return "\n".join(lines) + "\n"

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.header)
        writer.writerows(self.rows)
        return buffer.getvalue()


def _markdown_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _number(value: float) -> str:
    return f"{value:.3f}"


def _tool_order(tool: str) -> tuple[int, str]:
    return tool.count("+"), tool


def load_records(
    results_dir: Path, split: DatasetSplit, coreference: bool = False
) -> dict[tuple[str, str], ResultRecord]:
    """The results of the split with co-reference off (or on), keyed by (tool, dataset)."""
    records = {}
    for path in sorted(results_dir.glob(RESULT_FILE_PATTERN)):
        record = read_record(path)
        if record.run.split is split and record.run.coreference is coreference:
            records[(record.run.tool, record.run.dataset)] = record
    return records


def load_runs(results_dir: Path, split: DatasetSplit) -> dict[tuple[str, str], EvaluationRun]:
    """The co-reference-off runs of the split, keyed by (tool, dataset)."""
    return {key: record.run for key, record in load_records(results_dir, split).items()}


def _tools(runs: dict[tuple[str, str], EvaluationRun], datasets: Sequence[str] = DATASETS) -> list[str]:
    return sorted({tool for tool, dataset in runs if dataset in datasets}, key=_tool_order)


def _reports(run: EvaluationRun) -> tuple[DetectionReport, DetectionReport]:
    return run.exact_boundaries, run.any_overlap


def table1(runs: dict[tuple[str, str], EvaluationRun]) -> Table:
    header = ["Tool"]
    for dataset in DATASETS:
        for rule in ("strict", "relaxed"):
            header += [f"{DATASET_TITLES[dataset]} P {rule}", f"{DATASET_TITLES[dataset]} R {rule}", f"{DATASET_TITLES[dataset]} F2 {rule}"]
    rows = []
    for tool in _tools(runs):
        row = [tool]
        for dataset in DATASETS:
            run = runs.get((tool, dataset))
            for report in _reports(run) if run else (None, None):
                score = report.overall_micro if report else None
                row += [_number(score.precision), _number(score.recall), _number(score.f2)] if score else [""] * 3
        rows.append(tuple(row))
    return Table("table1", "Table 1: Detection by tool, overall (micro)", tuple(header), tuple(rows))


def _type_cell(report: DetectionReport, pii_type: PiiType, dataset: str) -> str:
    if pii_type is PiiType.IDENTIFIER and dataset != "tab":
        return ""
    score = report.by_type.get(pii_type)
    return _number(score.f2) if score else NOT_APPLICABLE


def table2(runs: dict[tuple[str, str], EvaluationRun], rule: SpanMatchRule) -> Table:
    variant = "strict" if rule is SpanMatchRule.EXACT_BOUNDARIES else "relaxed"
    header = ("Dataset", "Tool", *(pii_type.name for pii_type in PiiType))
    rows = []
    for dataset in DATASETS:
        for tool in _tools(runs, (dataset,)):
            run = runs.get((tool, dataset))
            if run is None:
                continue
            report = run.exact_boundaries if rule is SpanMatchRule.EXACT_BOUNDARIES else run.any_overlap
            rows.append((DATASET_TITLES[dataset], tool, *(_type_cell(report, t, dataset) for t in PiiType)))
    return Table(f"table2_{variant}", f"Table 2: F2 by PII type, {variant} matching", header, tuple(rows))


def table3(runs: dict[tuple[str, str], EvaluationRun]) -> Table:
    header = ("Tool", "ER_di", "ER_qi", "WP", "Any-annotator F2 strict", "Any-annotator F2 relaxed")
    rows = []
    for tool in _tools(runs, ("tab",)):
        run = runs.get((tool, "tab"))
        if run is None or run.tab_metrics is None:
            continue
        metrics = run.tab_metrics
        rows.append(
            (
                tool,
                _number(metrics.er_di),
                _number(metrics.er_qi),
                _number(metrics.wp),
                _number(run.exact_boundaries.overall_micro.f2),
                _number(run.any_overlap.overall_micro.f2),
            )
        )
    return Table("table3", "Table 3: TAB official metrics", header, tuple(rows))


def _change(off: float, on: float) -> str:
    return f"{on - off:+.3f}"


def table4(off: dict[tuple[str, str], ResultRecord], on: dict[tuple[str, str], ResultRecord]) -> Table:
    """F2 and TAB entity-level recall with co-reference off, on and the change; English text only.

    MAPA rows use the English figures of the run. MAPA has no entity ids, so its recall cells are blank.
    """
    header = (
        "Dataset", "Detector", "F2 off", "F2 on", "F2 change",
        "ER_di off", "ER_di on", "ER_di change", "ER_qi off", "ER_qi on", "ER_qi change",
    )
    rows = []
    for dataset in DATASETS:
        for tool in sorted({tool for tool, name in off if name == dataset}, key=_tool_order):
            if (tool, dataset) not in on:
                continue
            before, after = off[(tool, dataset)].run, on[(tool, dataset)].run
            if dataset == "mapa":
                scores = [r.exact_boundaries.by_language.get(ENGLISH) for r in (before, after)]
                if None in scores:
                    continue
                f2_off, f2_on = scores[0].f2, scores[1].f2
            else:
                f2_off, f2_on = before.exact_boundaries.overall_micro.f2, after.exact_boundaries.overall_micro.f2
            recall = ["", "", "", "", "", ""]
            if before.tab_metrics and after.tab_metrics:
                recall = [
                    _number(before.tab_metrics.er_di), _number(after.tab_metrics.er_di),
                    _change(before.tab_metrics.er_di, after.tab_metrics.er_di),
                    _number(before.tab_metrics.er_qi), _number(after.tab_metrics.er_qi),
                    _change(before.tab_metrics.er_qi, after.tab_metrics.er_qi),
                ]
            rows.append((DATASET_TITLES[dataset], tool, _number(f2_off), _number(f2_on), _change(f2_off, f2_on), *recall))
    return Table("table4", "Table 4: Co-reference off versus on (English text, strict matching)", header, tuple(rows))


HEADLINE_LANGUAGES = ("en", "fr", "de")
TAB_REFERENCE_COLUMN = "TAB (English reference)"


def _language_cell(record: ResultRecord | None, language: str, f2: float | None) -> str:
    """n/a when the detector does not support the language; blank when there is no run."""
    if record is None:
        return ""
    if record.languages is not None and language not in record.languages:
        return NOT_APPLICABLE
    return "" if f2 is None else _number(f2)


def table5(records: dict[tuple[str, str], ResultRecord], rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES) -> Table:
    """F2 per MAPA language, headline languages first, with TAB as the English reference column."""
    def report(record: ResultRecord) -> DetectionReport:
        return record.run.exact_boundaries if rule is SpanMatchRule.EXACT_BOUNDARIES else record.run.any_overlap

    mapa = {tool: record for (tool, dataset), record in records.items() if dataset == "mapa"}
    seen = {language for record in mapa.values() for language in report(record).by_language}
    seen |= {language for record in mapa.values() for language in (record.languages or ())}
    languages = [*HEADLINE_LANGUAGES, *sorted(seen - set(HEADLINE_LANGUAGES))]
    header = ("Tool", *languages, TAB_REFERENCE_COLUMN)
    rows = []
    for tool in sorted({tool for tool, _ in records}, key=_tool_order):
        cells = []
        for language in languages:
            record = mapa.get(tool)
            score = report(record).by_language.get(language) if record else None
            cells.append(_language_cell(record, language, score.f2 if score else None))
        tab_record = records.get((tool, "tab"))
        tab_f2 = report(tab_record).overall_micro.f2 if tab_record else None
        rows.append((tool, *cells, _language_cell(tab_record, "en", tab_f2)))
    variant = "strict" if rule is SpanMatchRule.EXACT_BOUNDARIES else "relaxed"
    return Table("table5", f"Table 5: F2 by language, {variant} matching (MAPA; TAB as the English reference)", header, tuple(rows))


def _write(tables: Sequence[Table], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for table in tables:
        for suffix, content in ((".md", table.to_markdown()), (".csv", table.to_csv())):
            path = out_dir / f"{table.name}{suffix}"
            path.write_text(content, encoding="utf-8", newline="")
            written.append(path)
    return written


def build_table1(results_dir: Path, out_dir: Path, split: DatasetSplit = DatasetSplit.TEST) -> list[Path]:
    return _write([table1(load_runs(results_dir, split))], out_dir)


def build_table2(results_dir: Path, out_dir: Path, split: DatasetSplit = DatasetSplit.TEST) -> list[Path]:
    runs = load_runs(results_dir, split)
    return _write([table2(runs, SpanMatchRule.EXACT_BOUNDARIES), table2(runs, SpanMatchRule.ANY_OVERLAP)], out_dir)


def build_table3(results_dir: Path, out_dir: Path, split: DatasetSplit = DatasetSplit.TEST) -> list[Path]:
    return _write([table3(load_runs(results_dir, split))], out_dir)


def build_table5(
    results_dir: Path,
    out_dir: Path,
    split: DatasetSplit = DatasetSplit.TEST,
    rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES,
) -> list[Path]:
    return _write([table5(load_records(results_dir, split), rule)], out_dir)


def build_table4(results_dir: Path, out_dir: Path, split: DatasetSplit = DatasetSplit.TEST) -> list[Path]:
    return _write([table4(load_records(results_dir, split), load_records(results_dir, split, coreference=True))], out_dir)
