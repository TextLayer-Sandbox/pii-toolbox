import re
import shutil
from pathlib import Path

from pii_toolbox.evaluation.charts import build_chart_a

RESULTS = Path(__file__).parent / "fixtures" / "results"
TOOLS = ("presidio_structured", "spacy_lg")


def _english_files(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    for tool in TOOLS:
        shutil.copy(
            RESULTS / f"{tool}__tab__test__coref-off.json",
            results / f"{tool}__tab__en__test__coref-off.json",
        )
    return results


def test_chart_a_plots_one_point_per_tool_from_english_files(tmp_path):
    svg = build_chart_a(_english_files(tmp_path), tmp_path / "out").read_text()
    texts = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
    for tool in TOOLS:
        assert texts.count(tool) == 1
