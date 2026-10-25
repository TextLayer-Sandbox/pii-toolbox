"""Charts of the article, drawn from saved result files only, as reproducible SVG files.

Matplotlib runs on the Agg backend with a fixed hash salt, no embedded date and text kept
as text, so two builds of the same results are byte identical and labels can be read from
the file.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pii_toolbox.core import DatasetSplit, DetectionReport, EvaluationRun, PiiType, SpanMatchRule  # noqa: E402
from pii_toolbox.evaluation.tables import (  # noqa: E402
    HEADLINE_LANGUAGES,
    tool_order,
    load_cost_results,
    load_records,
    load_runs,
)

F2_CONTOUR_LEVELS = (0.5, 0.7, 0.9)
SVG_HASH_SALT = "pii-toolbox"
CHART_A_FILE_NAME = "chart_a.svg"
_CURVE_POINTS = 200
MEMORY_MARKER_AREA_PER_GIB = 300


def _svg_settings() -> dict[str, object]:
    return {"svg.hashsalt": SVG_HASH_SALT, "svg.fonttype": "none", "font.family": "DejaVu Sans"}


def _f2_contour(level: float) -> tuple[np.ndarray, np.ndarray]:
    """Recall as a function of precision along the points where F2 = 5PR/(4P+R) equals level."""
    lowest_precision = level / (5 - 4 * level)
    precision = np.linspace(lowest_precision, 1.0, _CURVE_POINTS)
    recall = 4 * level * precision / (5 * precision - level)
    return precision, recall


def chart_a_figure(runs: Sequence[EvaluationRun], rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES) -> Figure:
    """Recall against precision, one labelled point per run, with F2 contour lines."""
    with plt.rc_context(_svg_settings()):
        figure, axes = plt.subplots(figsize=(7, 6))
        for level in F2_CONTOUR_LEVELS:
            precision, recall = _f2_contour(level)
            label = f"F2={level}"
            axes.plot(precision, recall, color="grey", linestyle="--", linewidth=1, label=label, gid=f"contour-{level}")
            axes.text(precision[-1], recall[-1], label, color="grey", fontsize=8, ha="right", va="bottom")
        for run in sorted(runs, key=lambda item: item.tool):
            report = run.exact_boundaries if rule is SpanMatchRule.EXACT_BOUNDARIES else run.any_overlap
            score = report.overall_micro
            axes.scatter([score.precision], [score.recall], s=40, zorder=3, label=run.tool, gid=f"point-{run.tool}")
            axes.annotate(run.tool, (score.precision, score.recall), textcoords="offset points", xytext=(5, 5), fontsize=8)
        axes.set_xlim(0, 1.02)
        axes.set_ylim(0, 1.02)
        axes.set_xlabel("Precision")
        axes.set_ylabel("Recall")
        axes.set_title(f"Chart A: recall against precision ({rule.name.lower().replace('_', ' ')})")
        axes.grid(True, linewidth=0.3)
    return figure


def _report(run: EvaluationRun, rule: SpanMatchRule) -> DetectionReport:
    return run.exact_boundaries if rule is SpanMatchRule.EXACT_BOUNDARIES else run.any_overlap


def _rule_name(rule: SpanMatchRule) -> str:
    return "strict" if rule is SpanMatchRule.EXACT_BOUNDARIES else "relaxed"


def _runs_of(results_dir: Path, dataset: str, split: DatasetSplit, coreference: bool = False) -> list[EvaluationRun]:
    records = load_records(results_dir, split, coreference)
    runs = [record.run for record in records.values() if record.run.dataset == dataset]
    return sorted(runs, key=lambda run: tool_order(run.tool))


def _save(figure: Figure, out_dir: Path, file_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / file_name
    with plt.rc_context(_svg_settings()):
        figure.savefig(path, format="svg", metadata={"Date": None, "Creator": "pii-toolbox"})
    plt.close(figure)
    return path


def _grouped_bars(
    axes, groups: Sequence[str], series: Sequence[str], value, id_prefix: str
) -> None:
    """Draws one bar per (group, series) side by side; `value(group, series)` gives its height."""
    width = 0.8 / max(len(series), 1)
    for index, name in enumerate(series):
        positions = [group_index + (index - (len(series) - 1) / 2) * width for group_index in range(len(groups))]
        container = axes.bar(positions, [value(group, name) for group in groups], width, label=name)
        for patch, group in zip(container.patches, groups):
            patch.set_gid(f"{id_prefix}-{group}-{name}")
    axes.set_xticks(range(len(groups)), groups)
    axes.legend(fontsize=8)


def chart_b_figure(runs: Sequence[EvaluationRun], rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES) -> Figure:
    """F2 by PII type: one group per type, one bar per tool; a type the tool does not report draws at zero."""
    with plt.rc_context(_svg_settings()):
        figure, axes = plt.subplots(figsize=(9, 5))
        by_tool = {run.tool: _report(run, rule).by_type for run in runs}

        def f2(pii_type: str, tool: str) -> float:
            score = by_tool[tool].get(PiiType[pii_type])
            return score.f2 if score else 0.0

        _grouped_bars(axes, [t.name for t in PiiType], list(by_tool), f2, "bar")
        axes.set_ylim(0, 1)
        axes.set_ylabel("F2")
        axes.set_title(f"Chart B: F2 by PII type ({_rule_name(rule)})")
    return figure


def chart_c_figure(runs: Sequence[EvaluationRun]) -> Figure:
    """Strict against relaxed overall F2 per tool; the gap is the boundary errors."""
    with plt.rc_context(_svg_settings()):
        figure, axes = plt.subplots(figsize=(8, 5))
        by_tool = {run.tool: run for run in runs}

        def f2(tool: str, rule_name: str) -> float:
            rule = SpanMatchRule.EXACT_BOUNDARIES if rule_name == "strict" else SpanMatchRule.ANY_OVERLAP
            return _report(by_tool[tool], rule).overall_micro.f2

        _grouped_bars(axes, list(by_tool), ["strict", "relaxed"], f2, "bar")
        axes.set_ylim(0, 1)
        axes.set_ylabel("F2 (overall, pooled)")
        axes.set_title("Chart C: strict versus relaxed matching")
        axes.tick_params(axis="x", labelrotation=20)
    return figure


def chart_d_figure(
    runs: Sequence[EvaluationRun], cost_results: Sequence[dict], rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES
) -> Figure:
    """F2 against latency per 1,000 characters, marker area proportional to peak memory."""
    costs = {result["setup"]: result for result in cost_results}
    with plt.rc_context(_svg_settings()):
        figure, axes = plt.subplots(figsize=(7, 6))
        for run in runs:
            cost = costs.get(run.tool)
            if cost is None:
                continue
            gib = cost["peak_memory_bytes"] / 1024**3
            axes.scatter(
                [cost["latency_per_1000_chars_median_seconds"]],
                [_report(run, rule).overall_micro.f2],
                s=MEMORY_MARKER_AREA_PER_GIB * gib,
                alpha=0.6,
                label=run.tool,
                gid=f"point-{run.tool}",
            )
            axes.annotate(run.tool, (cost["latency_per_1000_chars_median_seconds"], _report(run, rule).overall_micro.f2), textcoords="offset points", xytext=(6, 6), fontsize=8)
        axes.set_xscale("log")
        axes.set_ylim(0, 1)
        axes.set_xlabel("Latency per 1,000 characters (s, log scale)")
        axes.set_ylabel("F2")
        axes.set_title(f"Chart D: accuracy against cost ({_rule_name(rule)}; marker area = peak memory)")
    return figure


def chart_e_figure(runs: Sequence[EvaluationRun], rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES) -> Figure:
    """F2 across languages, one line per tool; a language a tool was not scored on is a gap."""
    seen = {language for run in runs for language in _report(run, rule).by_language}
    languages = [*HEADLINE_LANGUAGES, *sorted(seen - set(HEADLINE_LANGUAGES))]
    with plt.rc_context(_svg_settings()):
        figure, axes = plt.subplots(figsize=(8, 5))
        for run in runs:
            by_language = _report(run, rule).by_language
            values = [by_language[language].f2 if language in by_language else np.nan for language in languages]
            axes.plot(range(len(languages)), values, marker="o", label=run.tool, gid=f"line-{run.tool}")
        axes.set_xticks(range(len(languages)), languages)
        axes.set_ylim(0, 1)
        axes.set_xlabel("Language")
        axes.set_ylabel("F2")
        axes.set_title(f"Chart E: F2 by language ({_rule_name(rule)})")
        axes.legend(fontsize=8)
    return figure


def chart_f_figure(
    off_runs: Sequence[EvaluationRun], on_runs: Sequence[EvaluationRun], rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES
) -> Figure:
    """Overall F2 before and after co-reference for every detector measured both ways."""
    on_by_tool = {run.tool: run for run in on_runs}
    pairs = {run.tool: run for run in off_runs if run.tool in on_by_tool}
    with plt.rc_context(_svg_settings()):
        figure, axes = plt.subplots(figsize=(8, 5))

        def f2(tool: str, state: str) -> float:
            run = pairs[tool] if state == "off" else on_by_tool[tool]
            return _report(run, rule).overall_micro.f2

        _grouped_bars(axes, list(pairs), ["off", "on"], f2, "bar")
        axes.set_ylim(0, 1)
        axes.set_ylabel("F2 (overall, pooled)")
        axes.set_title(f"Chart F: co-reference effect ({_rule_name(rule)})")
        axes.tick_params(axis="x", labelrotation=20)
    return figure


def build_chart_b(results_dir: Path, out_dir: Path, dataset: str = "tab", split: DatasetSplit = DatasetSplit.TEST, rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES) -> Path:
    return _save(chart_b_figure(_runs_of(results_dir, dataset, split), rule), out_dir, "chart_b.svg")


def build_chart_c(results_dir: Path, out_dir: Path, dataset: str = "tab", split: DatasetSplit = DatasetSplit.TEST) -> Path:
    return _save(chart_c_figure(_runs_of(results_dir, dataset, split)), out_dir, "chart_c.svg")


def build_chart_d(results_dir: Path, cost_dir: Path, out_dir: Path, dataset: str = "tab", split: DatasetSplit = DatasetSplit.TEST, rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES) -> Path:
    figure = chart_d_figure(_runs_of(results_dir, dataset, split), load_cost_results(cost_dir), rule)
    return _save(figure, out_dir, "chart_d.svg")


def build_chart_e(results_dir: Path, out_dir: Path, dataset: str = "mapa", split: DatasetSplit = DatasetSplit.TEST, rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES) -> Path:
    return _save(chart_e_figure(_runs_of(results_dir, dataset, split), rule), out_dir, "chart_e.svg")


def build_chart_f(results_dir: Path, out_dir: Path, dataset: str = "tab", split: DatasetSplit = DatasetSplit.TEST, rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES) -> Path:
    figure = chart_f_figure(_runs_of(results_dir, dataset, split), _runs_of(results_dir, dataset, split, True), rule)
    return _save(figure, out_dir, "chart_f.svg")


def build_chart_a(
    results_dir: Path,
    out_dir: Path,
    dataset: str = "tab",
    split: DatasetSplit = DatasetSplit.TEST,
    rule: SpanMatchRule = SpanMatchRule.EXACT_BOUNDARIES,
) -> Path:
    """Writes chart_a.svg for the co-reference-off runs of one dataset and split."""
    runs = [run for (_, name), run in load_runs(results_dir, split).items() if name == dataset]
    figure = chart_a_figure(runs, rule)
    return _save(figure, out_dir, CHART_A_FILE_NAME)
