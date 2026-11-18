"""The evaluation tables 1 to 6, built from saved result files only.

Each builder reads the result JSON files of one split with co-reference off and writes a
Markdown and a CSV file per table. Rows are tools, single detectors first and then
combinations, each in name order; numbers print with three decimals, so the output is
the same on every run.
"""

import csv
import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pii_toolbox.core import DatasetSplit, DetectionReport, DetectionScore, EvaluationRun, PiiType, SpanMatchRule
from pii_toolbox.datasets.mapa import MapaDataset
from pii_toolbox.datasets.tab import TabDataset
from pii_toolbox.datasets.worldwide import WorldwideDataset
from pii_toolbox.evaluation.results_io import ResultRecord, read_record

DATASETS = ("tab", "mapa", "worldwide")
DATASET_TITLES = {"tab": "TAB", "mapa": "MAPA", "worldwide": "Worldwide"}
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


def tool_order(tool: str) -> tuple[int, str]:
    return tool.count("+"), tool


def load_records(
    results_dir: Path, split: DatasetSplit, coreference: bool = False
) -> dict[tuple[str, str, str | None], ResultRecord]:
    """The results of the split with co-reference off (or on), keyed by (tool, dataset, language).

    The language is None for a file that covers every language of its dataset.
    """
    records = {}
    for path in sorted(results_dir.glob(RESULT_FILE_PATTERN)):
        record = read_record(path)
        if record.run.split is split and record.run.coreference is coreference:
            records[(record.run.tool, record.run.dataset, record.language)] = record
    return records


def load_runs(results_dir: Path, split: DatasetSplit) -> dict[tuple[str, str], EvaluationRun]:
    """The co-reference-off runs of the split that cover a whole dataset, keyed by (tool, dataset)."""
    return {
        (tool, dataset): record.run
        for (tool, dataset, language), record in load_records(results_dir, split).items()
        if language is None
    }


def english_runs(
    records: dict[tuple[str, str, str | None], ResultRecord],
) -> dict[tuple[str, str], EvaluationRun]:
    """The runs holding each tool's English figures per dataset: the whole-dataset file, else the English-only file."""
    runs = {}
    for tool, dataset in sorted({(tool, dataset) for tool, dataset, _ in records}):
        record = _english_record(records, tool, dataset)
        if record is not None:
            runs[(tool, dataset)] = record.run
    return runs


def _tools(runs: dict[tuple[str, str], EvaluationRun], datasets: Sequence[str] = DATASETS) -> list[str]:
    return sorted({tool for tool, dataset in runs if dataset in datasets}, key=tool_order)


def _rule_report(run: EvaluationRun, rule: SpanMatchRule) -> DetectionReport:
    return run.exact_boundaries if rule is SpanMatchRule.EXACT_BOUNDARIES else run.any_overlap


def _reports(run: EvaluationRun) -> tuple[DetectionReport, DetectionReport]:
    return run.exact_boundaries, run.any_overlap


def _detects_nothing_annotated(record: ResultRecord, dataset: str) -> bool:
    """True when the detector can report none of the types the dataset annotates, so its scores carry no information."""
    dataset_classes = {"tab": TabDataset, "mapa": MapaDataset, "worldwide": WorldwideDataset}
    return not record.detects & dataset_classes[dataset]().annotated_types


def _english_record(
    records: dict[tuple[str, str, str | None], ResultRecord], tool: str, dataset: str
) -> ResultRecord | None:
    """The run holding the English figures: the whole-dataset file, else the English-only file."""
    return records.get((tool, dataset, None)) or records.get((tool, dataset, ENGLISH))


def table1(records: dict[tuple[str, str, str | None], ResultRecord]) -> Table:
    """Overall detection by tool; the MAPA columns are the English scores because MAPA is measured per language."""
    def label(dataset: str) -> str:
        return f"{DATASET_TITLES[dataset]} ({ENGLISH})" if dataset == "mapa" else DATASET_TITLES[dataset]

    def scores(tool: str, dataset: str, report_of: Callable[[EvaluationRun], DetectionReport]) -> DetectionScore | None:
        record = _english_record(records, tool, dataset)
        if record is None:
            return None
        report = report_of(record.run)
        return report.by_language.get(ENGLISH) if dataset == "mapa" else report.overall_micro

    header = ["Tool"]
    for dataset in DATASETS:
        for rule in ("strict", "relaxed"):
            header += [f"{label(dataset)} P {rule}", f"{label(dataset)} R {rule}", f"{label(dataset)} F2 {rule}"]
    rows = []
    tools = sorted({tool for tool, dataset, _ in records if dataset in DATASETS}, key=tool_order)
    for tool in tools:
        row = [tool]
        for dataset in DATASETS:
            for report_of in (lambda run: run.exact_boundaries, lambda run: run.any_overlap):
                score = scores(tool, dataset, report_of)
                record = _english_record(records, tool, dataset)
                if record is not None and _detects_nothing_annotated(record, dataset):
                    row += [NOT_APPLICABLE] * 3
                else:
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


def table4(
    off: dict[tuple[str, str, str | None], ResultRecord],
    on: dict[tuple[str, str, str | None], ResultRecord],
    rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES,
) -> Table:
    """F2 and TAB entity-level recall with co-reference off, on and the change; English text only.

    MAPA rows use the English figures of the run. MAPA has no entity ids, so its recall cells are blank.
    The F2 columns follow `rule`; the entity-level recall columns are TAB's own metric.
    """
    header = (
        "Dataset", "Detector", "F2 off", "F2 on", "F2 change",
        "ER_di off", "ER_di on", "ER_di change", "ER_qi off", "ER_qi on", "ER_qi change",
    )
    rows = []
    for dataset in DATASETS:
        for tool in sorted({tool for tool, name, _ in off if name == dataset}, key=tool_order):
            before_record, after_record = _english_record(off, tool, dataset), _english_record(on, tool, dataset)
            if before_record is None or after_record is None:
                continue
            before, after = before_record.run, after_record.run
            if dataset == "mapa":
                scores = [_rule_report(r, rule).by_language.get(ENGLISH) for r in (before, after)]
                if None in scores:
                    continue
                f2_off, f2_on = scores[0].f2, scores[1].f2
            else:
                f2_off, f2_on = _rule_report(before, rule).overall_micro.f2, _rule_report(after, rule).overall_micro.f2
            recall = ["", "", "", "", "", ""]
            if before.tab_metrics and after.tab_metrics:
                recall = [
                    _number(before.tab_metrics.er_di), _number(after.tab_metrics.er_di),
                    _change(before.tab_metrics.er_di, after.tab_metrics.er_di),
                    _number(before.tab_metrics.er_qi), _number(after.tab_metrics.er_qi),
                    _change(before.tab_metrics.er_qi, after.tab_metrics.er_qi),
                ]
            rows.append((DATASET_TITLES[dataset], tool, _number(f2_off), _number(f2_on), _change(f2_off, f2_on), *recall))
    variant = "strict" if rule is SpanMatchRule.EXACT_BOUNDARIES else "relaxed"
    return Table(
        "table4", f"Table 4: Co-reference off versus on (English text, {variant} matching)", header, tuple(rows)
    )


LAMBDA_MEMORY_BYTES = 10 * 1024**3
COST_INPUT_SIZES = ("1000", "10000", "100000")
_BYTES_PER_GIB = 1024**3


def load_cost_results(cost_dir: Path) -> list[dict]:
    """The saved cost measurements, one per setup, ordered single tools first then combinations."""
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(cost_dir.glob("*.json"))]
    return sorted(results, key=lambda result: tool_order(result["setup"]))


def table6(results: Sequence[dict]) -> Table:
    """Peak memory, load time, throughput, latency and whether the setup fits AWS Lambda's 10 GB."""
    header = (
        "Tool",
        "Peak memory (GiB)",
        "Load time (s)",
        *(f"Throughput at {int(size) // 1000}k chars (chars/s)" for size in COST_INPUT_SIZES),
        "Latency per 1,000 chars (s)",
        "fits Lambda (10 GB)",
    )
    rows = []
    for result in results:
        peak = result["peak_memory_bytes"]
        rows.append(
            (
                result["setup"],
                f"{peak / _BYTES_PER_GIB:.2f}",
                f"{result['load_seconds']:.2f}",
                *(f"{result['throughput_chars_per_second'][size]:.0f}" for size in COST_INPUT_SIZES),
                f"{result['latency_per_1000_chars_median_seconds']:.3f}",
                "yes" if peak < LAMBDA_MEMORY_BYTES else "no",
            )
        )
    return Table("table6", "Table 6: Cost and speed", header, tuple(rows))


HEADLINE_LANGUAGES = ("en", "fr", "de")
TAB_REFERENCE_COLUMN = "TAB (English reference)"


def _language_cell(record: ResultRecord | None, language: str, f2: float | None) -> str:
    """n/a when the detector does not support the language; blank when there is no run."""
    if record is None:
        return ""
    if record.languages is not None and language not in record.languages:
        return NOT_APPLICABLE
    return "" if f2 is None else _number(f2)


def table5(
    records: dict[tuple[str, str, str | None], ResultRecord], rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES
) -> Table:
    """F2 per MAPA language, headline languages first, with TAB as the English reference column.

    A language comes from the file measured on it, or from the by-language scores of a file that covers
    every language.
    """
    def report(record: ResultRecord) -> DetectionReport:
        return record.run.exact_boundaries if rule is SpanMatchRule.EXACT_BOUNDARIES else record.run.any_overlap

    mapa: dict[str, list[ResultRecord]] = {}
    for (tool, dataset, _), record in records.items():
        if dataset == "mapa":
            mapa.setdefault(tool, []).append(record)
    seen = set()
    for tool_records in mapa.values():
        for record in tool_records:
            seen |= set(report(record).by_language) | set(record.languages or ())
            if record.language is not None:
                seen.add(record.language)
    languages = [*HEADLINE_LANGUAGES, *sorted(seen - set(HEADLINE_LANGUAGES))]
    header = ("Tool", *languages, TAB_REFERENCE_COLUMN)
    rows = []
    for tool in sorted({tool for tool, _, _ in records}, key=tool_order):
        cells = []
        for language in languages:
            tool_records = mapa.get(tool, [])
            measured = next((r for r in tool_records if r.language == language), None) or next(
                (r for r in tool_records if r.language is None), None
            )
            score = report(measured).by_language.get(language) if measured else None
            cell_record = measured or next(iter(tool_records), None)
            if cell_record is not None and _detects_nothing_annotated(cell_record, "mapa"):
                cells.append(NOT_APPLICABLE)
            else:
                cells.append(_language_cell(cell_record, language, score.f2 if score else None))
        tab_record = _english_record(records, tool, "tab")
        tab_f2 = report(tab_record).overall_micro.f2 if tab_record else None
        rows.append((tool, *cells, _language_cell(tab_record, "en", tab_f2)))
    variant = "strict" if rule is SpanMatchRule.EXACT_BOUNDARIES else "relaxed"
    return Table("table5", f"Table 5: F2 by language, {variant} matching (MAPA; TAB as the English reference)", header, tuple(rows))


def write_tables(tables: Sequence[Table], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for table in tables:
        for suffix, content in ((".md", table.to_markdown()), (".csv", table.to_csv())):
            path = out_dir / f"{table.name}{suffix}"
            path.write_text(content, encoding="utf-8", newline="")
            written.append(path)
    return written

