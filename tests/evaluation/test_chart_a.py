import re
from pathlib import Path

import pytest

from pii_toolbox.core import DatasetSplit, SpanMatchRule
from pii_toolbox.evaluation.charts import build_chart_a, chart_a_figure
from pii_toolbox.evaluation.tables import load_runs

RESULTS = Path(__file__).parent / "fixtures" / "results"


def tab_runs():
    return [run for (_, dataset), run in load_runs(RESULTS, DatasetSplit.TEST).items() if dataset == "tab"]


def svg_texts(path):
    return re.findall(r"<text[^>]*>([^<]*)</text>", path.read_text())


def test_chart_has_one_labelled_point_per_setup_and_three_labelled_contours(tmp_path):
    texts = svg_texts(build_chart_a(RESULTS, tmp_path))
    for tool in ("presidio_structured", "spacy_lg", "spacy_lg+presidio_structured"):
        assert texts.count(tool) == 1
    for label in ("F2=0.5", "F2=0.7", "F2=0.9"):
        assert label in texts


def test_points_lie_at_precision_and_recall_from_the_fixture():
    runs = tab_runs()
    axes = chart_a_figure(runs).axes[0]
    drawn = {tuple(map(float, offset)) for collection in axes.collections for offset in collection.get_offsets()}
    expected = {(run.exact_boundaries.overall_micro.precision, run.exact_boundaries.overall_micro.recall) for run in runs}
    assert drawn == expected


def test_relaxed_rule_plots_relaxed_scores():
    runs = tab_runs()
    axes = chart_a_figure(runs, SpanMatchRule.ANY_OVERLAP).axes[0]
    drawn = {tuple(map(float, offset)) for collection in axes.collections for offset in collection.get_offsets()}
    assert drawn == {(run.any_overlap.overall_micro.precision, run.any_overlap.overall_micro.recall) for run in runs}


def test_contours_are_f2_level_sets():
    axes = chart_a_figure(tab_runs()).axes[0]
    contours = {line.get_label(): line for line in axes.lines}
    for level in (0.5, 0.7, 0.9):
        precision, recall = contours[f"F2={level}"].get_data()
        f2 = 5 * precision * recall / (4 * precision + recall)
        assert f2 == pytest.approx(level)


def test_two_builds_are_byte_identical(tmp_path):
    first = build_chart_a(RESULTS, tmp_path / "one").read_bytes()
    second = build_chart_a(RESULTS, tmp_path / "two").read_bytes()
    assert first == second
    assert b"<dc:date>" not in first

