import pytest

from pii_toolbox.core import Detector, DetectorRegistry
from pii_toolbox.detection.registry import FixedDetectorRegistry, UnknownDetectorError
from pii_toolbox.evaluation import configs
from pii_toolbox.evaluation.configs import all_runs, measured_configs, model_tools

REAL_MODEL_TOOLS = {"gliner", "roberta", "spacy_lg", "spacy_sm", "llm"}


class NamesOnlyRegistry(DetectorRegistry):
    def __init__(self, names):
        self._names = frozenset(names)

    def names(self):
        return self._names

    def build(self, name) -> Detector:
        raise AssertionError("configurations are listed without building any detector")


def test_model_tools_are_the_registry_names_that_are_not_pattern_or_list_detectors():
    assert set(model_tools()) == REAL_MODEL_TOOLS


def test_configs_are_presidio_alone_each_model_alone_and_each_model_with_presidio():
    result = measured_configs()
    n = len(REAL_MODEL_TOOLS)
    assert len(result) == 2 * n + 1 == 11
    assert len(set(result)) == len(result)
    assert result[0] == "presidio_structured"
    assert set(result[1 : n + 1]) == REAL_MODEL_TOOLS
    assert set(result[n + 1 :]) == {f"{tool}+presidio_structured" for tool in REAL_MODEL_TOOLS}


def test_there_are_no_model_pairs_and_no_union_of_all_models():
    for config in measured_configs():
        members = config.split("+")
        assert len(members) <= 2
        assert sum(member in REAL_MODEL_TOOLS for member in members) <= 1


def test_every_config_name_validates_against_the_registry():
    registry = FixedDetectorRegistry()
    configs.validate_configs(measured_configs(registry), registry)
    with pytest.raises(UnknownDetectorError):
        configs.validate_configs(["spacy_lg+typo"], registry)


def test_the_list_is_generated_from_the_registry_not_typed_out():
    registry = NamesOnlyRegistry({"presidio_structured", "known_names", "alpha", "beta"})
    assert measured_configs(registry) == [
        "presidio_structured",
        "alpha",
        "beta",
        "alpha+presidio_structured",
        "beta+presidio_structured",
    ]


def test_each_config_runs_twice_on_english_and_once_on_french_and_german():
    runs = list(all_runs())
    for config in measured_configs():
        mine = [(r.dataset, r.language, r.coreference) for r in runs if r.config == config]
        assert sorted(mine) == sorted(
            [
                ("tab", "en", False),
                ("tab", "en", True),
                ("mapa", "en", False),
                ("mapa", "en", True),
                ("mapa", "fr", False),
                ("mapa", "de", False),
            ]
        )
    assert len(runs) == 6 * 11
    assert len(set(runs)) == len(runs)


def test_co_reference_is_only_ever_on_for_english_text():
    assert all(run.language == "en" for run in all_runs() if run.coreference)
