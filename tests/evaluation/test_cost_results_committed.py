import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "cost"
GIB = 1024**3
INPUT_SIZES = ("1000", "10000", "100000")
LAMBDA_LIMIT_BYTES = 10 * GIB


def load_script():
    spec = importlib.util.spec_from_file_location("run_cost_all", REPO_ROOT / "scripts" / "run_cost_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SETUPS = load_script().measured_configs()


def result_path(setup):
    return RESULTS / f"{setup.replace('+', '__')}.json"


def test_setups_follow_the_agreed_list():
    assert "presidio_structured" in SETUPS
    models = [name for name in SETUPS if "+" not in name and name != "presidio_structured"]
    assert models
    for model in models:
        assert f"{model}+presidio_structured" in SETUPS
    for name in SETUPS:
        if "+" in name:
            assert name.split("+")[1] == "presidio_structured"
            assert name.count("+") == 1
    assert "llm" not in " ".join(SETUPS)


def test_only_listed_setups_have_result_files():
    expected = {result_path(setup).name for setup in SETUPS}
    assert {path.name for path in RESULTS.glob("*.json")} == expected


@pytest.mark.parametrize("setup", SETUPS)
def test_result_file_has_every_figure_and_its_environment(setup):
    result = json.loads(result_path(setup).read_text())
    assert result["setup"] == setup
    assert 0 < result["peak_memory_bytes"] < 32 * GIB
    assert result["load_seconds"] >= 0
    assert set(result["throughput_chars_per_second"]) == set(INPUT_SIZES)
    assert all(rate > 0 for rate in result["throughput_chars_per_second"].values())
    assert result["latency_per_1000_chars_median_seconds"] >= 0
    assert result["fits_lambda"] == (result["peak_memory_bytes"] < LAMBDA_LIMIT_BYTES)
    environment = result["environment"]
    assert {"machine", "os", "python", "library_versions"} <= set(environment)
    assert environment["library_versions"]
    assert "language model" in " ".join(result["not_measured"])
