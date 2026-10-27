"""The detector configurations the evaluation measures and the runs each one gets.

The configuration list is generated from the detector registry's names: Presidio's structured
recognisers alone, each model tool alone, and each model tool joined with Presidio's. There
are no unions of two models. Every configuration is run with co-reference off and on for
English text and with it off only for other languages, because the co-reference model is
English-only.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from pii_toolbox.core import DetectorRegistry
from pii_toolbox.detection.registry import UNION_SEPARATOR, FixedDetectorRegistry, validate_config_names

PATTERN_TOOL = "presidio_structured"
NON_MODEL_TOOLS = frozenset({PATTERN_TOOL, "known_names"})
ENGLISH = "en"
# (dataset, language, co-reference states measured)
RUN_PLAN: tuple[tuple[str, str, tuple[bool, ...]], ...] = (
    ("tab", ENGLISH, (False, True)),
    ("mapa", ENGLISH, (False, True)),
    ("worldwide", ENGLISH, (False, True)),
    ("mapa", "fr", (False,)),
    ("mapa", "de", (False,)),
)


@dataclass(frozen=True)
class RunSpec:
    """One evaluation run: a configuration on a dataset and language, with co-reference off or on."""

    config: str
    dataset: str
    language: str
    coreference: bool


def model_tools(registry: DetectorRegistry | None = None) -> list[str]:
    """The registry's model-backed detectors, in name order."""
    registry = registry or FixedDetectorRegistry()
    return sorted(registry.names() - NON_MODEL_TOOLS)


def measured_configs(registry: DetectorRegistry | None = None) -> list[str]:
    """Presidio alone, each model alone, and each model with Presidio, as `a` and `a+presidio_structured` names."""
    registry = registry or FixedDetectorRegistry()
    models = model_tools(registry)
    configs = [PATTERN_TOOL, *models, *(f"{model}{UNION_SEPARATOR}{PATTERN_TOOL}" for model in models)]
    validate_config_names(configs, registry)
    return configs


def all_runs(registry: DetectorRegistry | None = None) -> Iterator[RunSpec]:
    """Every run of every configuration: TAB, MAPA and Worldwide English with co-reference off and on, MAPA French and German off."""
    for config in measured_configs(registry):
        for dataset, language, states in RUN_PLAN:
            for coreference in states:
                yield RunSpec(config, dataset, language, coreference)
