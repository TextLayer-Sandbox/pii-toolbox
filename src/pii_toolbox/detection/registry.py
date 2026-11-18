"""The fixed table of detector names and the parser that turns a configuration into detectors."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pii_toolbox.core import Detector, DetectorRegistry
from pii_toolbox.detection.gliner_detector import GlinerDetector
from pii_toolbox.detection.known_names import KnownNamesDetector
from pii_toolbox.detection.presidio_structured import PresidioStructuredDetector
from pii_toolbox.detection.roberta_detector import RobertaDetector
from pii_toolbox.detection.spacy_detectors import spacy_lg_detector, spacy_sm_detector
from pii_toolbox.detection.union import MergingDetectorUnion

UNION_SEPARATOR = "+"

DetectorFactory = Callable[[], Detector]


class UnknownDetectorError(ValueError):
    """Raised when a configuration names an unknown detector."""


class FixedDetectorRegistry(DetectorRegistry):
    """Every detector the toolbox offers, built by name; `factory_overrides` replaces entries with other builders."""

    def __init__(
        self,
        model_dir: Path | None = None,
        known_names: Sequence[str] = (),
        known_names_case_insensitive: bool = False,
        factory_overrides: Mapping[str, DetectorFactory] | None = None,
    ) -> None:
        self._factories: dict[str, DetectorFactory] = {
            "presidio_structured": PresidioStructuredDetector,
            "spacy_lg": lambda: spacy_lg_detector(model_dir),
            "spacy_sm": lambda: spacy_sm_detector(model_dir),
            "roberta": lambda: RobertaDetector(model_dir),
            "gliner": lambda: GlinerDetector(model_dir),
            "known_names": lambda: KnownNamesDetector(known_names, known_names_case_insensitive),
        }
        self._factories.update(factory_overrides or {})

    def names(self) -> frozenset[str]:
        return frozenset(self._factories)

    def build(self, name: str) -> Detector:
        if name not in self._factories:
            known = ", ".join(sorted(self._factories))
            raise UnknownDetectorError(f"unknown detector {name!r}; known detectors: {known}")
        return self._factories[name]()


def validate_config_names(entries: Sequence[str], registry: DetectorRegistry) -> None:
    """Raises when any member of any entry is not in `registry`."""
    known = registry.names()
    for entry in entries:
        for name in entry.split(UNION_SEPARATOR):
            if name not in known:
                raise UnknownDetectorError(
                    f"unknown detector {name!r}; known detectors: {', '.join(sorted(known))}"
                )


def build_from_config(entries: Sequence[str], registry: DetectorRegistry | None = None) -> list[Detector]:
    """Builds one detector per entry, where an entry `a+b` builds a union of `a` and `b`; checks every name before building anything."""
    registry = registry or FixedDetectorRegistry()
    validate_config_names(entries, registry)
    detectors: list[Detector] = []
    for entry in entries:
        members = tuple(registry.build(name) for name in entry.split(UNION_SEPARATOR))
        detectors.append(members[0] if len(members) == 1 else MergingDetectorUnion(members))
    return detectors
