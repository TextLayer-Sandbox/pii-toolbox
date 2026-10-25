import re
from pathlib import Path

import pytest

from pii_toolbox.core import DatasetSplit, PiiType
from pii_toolbox.evaluation import charts
from pii_toolbox.evaluation.tables import load_cost_results

FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = FIXTURES / "results"
COREF_RESULTS = FIXTURES / "results_coref"
COST = FIXTURES / "cost_charts"
TAB_TOOLS = 3  # presidio_structured, spacy_lg, spacy_lg+presidio_structured
MAPA_TOOLS = 2  # spacy_lg, spacy_lg+presidio_structured


def ids(path, prefix):
    return re.findall(rf'id="({prefix}[^"]*)"', path.read_text())


def build_all(out_dir):
    return {
        "b": charts.build_chart_b(RESULTS, out_dir),
        "c": charts.build_chart_c(RESULTS, out_dir),
        "d": charts.build_chart_d(RESULTS, COST, out_dir),
        "e": charts.build_chart_e(RESULTS, out_dir),
        "f": charts.build_chart_f(COREF_RESULTS, out_dir),
    }


def test_chart_b_has_a_bar_per_type_and_tool(tmp_path):
    bars = ids(charts.build_chart_b(RESULTS, tmp_path), "bar-")
    assert len(bars) == len(PiiType) * TAB_TOOLS
    assert len(set(bars)) == len(bars)
    assert {bar.split("-")[1] for bar in bars} == {t.name for t in PiiType}


def test_chart_c_has_a_strict_and_a_relaxed_bar_per_tool(tmp_path):
    bars = ids(charts.build_chart_c(RESULTS, tmp_path), "bar-")
    assert len(bars) == 2 * TAB_TOOLS
    assert sum(bar.endswith("-strict") for bar in bars) == sum(bar.endswith("-relaxed") for bar in bars) == TAB_TOOLS


def test_chart_c_relaxed_bars_are_not_lower_than_strict():
    from pii_toolbox.evaluation.charts import chart_c_figure
    from pii_toolbox.evaluation.tables import load_runs

    runs = [run for (_, dataset), run in load_runs(RESULTS, DatasetSplit.TEST).items() if dataset == "tab"]
    heights = [patch.get_height() for patch in chart_c_figure(runs).axes[0].patches]
    strict, relaxed = heights[:TAB_TOOLS], heights[TAB_TOOLS:]
    assert all(r >= s for s, r in zip(strict, relaxed))


def test_chart_d_has_a_point_per_costed_tool_with_area_scaling_with_memory(tmp_path):
    assert len(ids(charts.build_chart_d(RESULTS, COST, tmp_path), "point-")) == TAB_TOOLS
    from pii_toolbox.evaluation.tables import load_runs

    runs = [run for (_, dataset), run in load_runs(RESULTS, DatasetSplit.TEST).items() if dataset == "tab"]
    axes = charts.chart_d_figure(runs, load_cost_results(COST)).axes[0]
    sizes = {c.get_label(): float(c.get_sizes()[0]) for c in axes.collections}
    assert sizes["spacy_lg"] == pytest.approx(3 * sizes["presidio_structured"])
    assert sizes["spacy_lg+presidio_structured"] == pytest.approx(4 * sizes["presidio_structured"])


def test_chart_d_leaves_out_a_tool_without_cost_results(tmp_path):
    only_one = tmp_path / "cost"
    only_one.mkdir()
    (only_one / "spacy_lg.json").write_text((COST / "spacy_lg.json").read_text())
    assert len(ids(charts.build_chart_d(RESULTS, only_one, tmp_path), "point-")) == 1


def test_chart_e_has_a_line_per_mapa_tool(tmp_path):
    assert len(ids(charts.build_chart_e(RESULTS, tmp_path), "line-")) == MAPA_TOOLS


def test_chart_f_pairs_off_and_on_for_detectors_measured_both_ways(tmp_path):
    bars = ids(charts.build_chart_f(COREF_RESULTS, tmp_path), "bar-")
    assert len(bars) == 2 * 2  # spacy_lg and gliner; presidio_structured has no co-reference-on run
    assert not [bar for bar in bars if "presidio" in bar]


def test_two_builds_of_every_chart_are_byte_identical(tmp_path):
    first = {name: path.read_bytes() for name, path in build_all(tmp_path / "one").items()}
    second = {name: path.read_bytes() for name, path in build_all(tmp_path / "two").items()}
    assert first == second
    assert all(b"<dc:date>" not in content for content in first.values())
