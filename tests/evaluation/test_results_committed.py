import json
from pathlib import Path

import pytest

from pii_toolbox.core import DatasetSplit
from pii_toolbox.evaluation.configs import all_runs, measured_configs
from pii_toolbox.evaluation.results_io import ENVIRONMENT_FILE_NAME, read_record

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "test"
RESULT_KEYS = {
    "tool",
    "coreference",
    "dataset",
    "split",
    "exact_boundaries",
    "any_overlap",
    "tab_metrics",
    "environment",
    "detector_languages",
    "detector_detects",
}
LONGEST_STRING = 200


def result_name(spec):
    return f"{spec.config}__{spec.dataset}__{spec.language}__test__coref-{'on' if spec.coreference else 'off'}.json"


def strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from strings(value)


RUNS = list(all_runs())


def test_every_run_in_the_plan_has_a_result_file_and_nothing_else_is_there():
    expected = {result_name(spec) for spec in RUNS} | {ENVIRONMENT_FILE_NAME}
    assert {path.name for path in RESULTS.glob("*.json")} == expected


@pytest.mark.parametrize("spec", RUNS, ids=result_name)
def test_result_file_matches_its_run(spec):
    run = read_record(RESULTS / result_name(spec)).run
    assert run.tool == spec.config
    assert run.dataset == spec.dataset
    assert run.split is DatasetSplit.TEST
    assert run.coreference is spec.coreference
    assert (run.tab_metrics is not None) == (spec.dataset == "tab")


def test_co_reference_off_and_on_both_exist_for_every_configuration_on_english_tab_mapa_and_worldwide():
    for config in measured_configs():
        for dataset in ("tab", "mapa", "worldwide"):
            for state in ("off", "on"):
                assert (RESULTS / f"{config}__{dataset}__en__test__coref-{state}.json").exists()


@pytest.mark.parametrize("spec", RUNS, ids=result_name)
def test_result_file_holds_only_aggregate_scores(spec):
    content = json.loads((RESULTS / result_name(spec)).read_text())
    assert set(content) == RESULT_KEYS
    assert all(len(text) <= LONGEST_STRING for text in strings(content))


def test_environment_file_describes_the_machine_and_holds_no_text():
    content = json.loads((RESULTS / ENVIRONMENT_FILE_NAME).read_text())
    assert set(content) == {"machine", "os", "runtime", "library_versions"}
    assert all(len(text) <= LONGEST_STRING for text in strings(content))
