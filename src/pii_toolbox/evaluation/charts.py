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

from pii_toolbox.core import DatasetSplit, EvaluationRun, SpanMatchRule  # noqa: E402
from pii_toolbox.evaluation.tables import load_runs  # noqa: E402

F2_CONTOUR_LEVELS = (0.5, 0.7, 0.9)
SVG_HASH_SALT = "pii-toolbox"
CHART_A_FILE_NAME = "chart_a.svg"
_CURVE_POINTS = 200


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
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CHART_A_FILE_NAME
    with plt.rc_context(_svg_settings()):
        figure.savefig(path, format="svg", metadata={"Date": None, "Creator": "pii-toolbox"})
    plt.close(figure)
    return path
