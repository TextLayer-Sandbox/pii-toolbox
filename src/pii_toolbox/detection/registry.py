"""The fixed table of detector names and the parser that turns a configuration into detectors."""

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pii_toolbox.core import Detector, DetectorRegistry
from pii_toolbox.detection.gliner_detector import GlinerDetector
from pii_toolbox.detection.known_names import KnownNamesDetector
from pii_toolbox.detection.llm_detector import LlmDetector, LlmRuntime
from pii_toolbox.detection.llm_runtimes import LlamaCppRuntime, MlxRuntime
from pii_toolbox.detection.presidio_structured import PresidioStructuredDetector
from pii_toolbox.detection.roberta_detector import RobertaDetector
from pii_toolbox.detection.spacy_detectors import spacy_lg_detector, spacy_sm_detector
from pii_toolbox.detection.union import DetectorUnion

LLM_MODEL_PATH_ENV_VAR = "PII_LLM_MODEL_PATH"
UNION_SEPARATOR = "+"

DetectorFactory = Callable[[], Detector]


class UnknownDetectorError(ValueError):
    """Raised when a configuration names a detector that is not in the registry."""


class _LoadOnDemandRuntime:
    """A language-model runtime that reads its model file only when first loaded or used."""

    def __init__(self, runtime_name: str, model_path: Path | None) -> None:
        self._runtime_name = runtime_name
        self._model_path = model_path
        self._runtime: LlmRuntime | None = None

    def load(self) -> None:
        if self._runtime is not None:
            return
        if self._model_path is None:
            raise FileNotFoundError(
                f"no language model configured: set llm_model_path or the {LLM_MODEL_PATH_ENV_VAR} environment variable"
            )
        runtime_class = {"llama_cpp": LlamaCppRuntime, "mlx": MlxRuntime}[self._runtime_name]
        self._runtime = runtime_class(self._model_path)

    def generate_structured(self, prompt: str, json_schema: dict[str, Any]) -> str:
        self.load()
        return self._runtime.generate_structured(prompt, json_schema)


class _ConfiguredLlmDetector(LlmDetector):
    """The language-model detector with a runtime that loads when `load_models` is called."""

    def __init__(self, runtime: _LoadOnDemandRuntime) -> None:
        super().__init__(runtime)
        self._load_on_demand_runtime = runtime

    def load_models(self) -> None:
        self._load_on_demand_runtime.load()


class FixedDetectorRegistry(DetectorRegistry):
    """Every detector the toolbox offers, built by name.

    Presidio's structured recognisers are in the table but only run when a configuration
    names `presidio_structured`. `factory_overrides` replaces entries, for tests that supply
    stand-in models.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        llm_model_path: Path | None = None,
        llm_runtime: str = "llama_cpp",
        known_names: Sequence[str] = (),
        known_names_case_insensitive: bool = False,
        factory_overrides: Mapping[str, DetectorFactory] | None = None,
    ) -> None:
        self._llm_model_path = llm_model_path
        self._factories: dict[str, DetectorFactory] = {
            "presidio_structured": PresidioStructuredDetector,
            "spacy_lg": lambda: spacy_lg_detector(model_dir),
            "spacy_sm": lambda: spacy_sm_detector(model_dir),
            "roberta": lambda: RobertaDetector(model_dir),
            "gliner": lambda: GlinerDetector(model_dir),
            "known_names": lambda: KnownNamesDetector(known_names, known_names_case_insensitive),
            "llm": lambda: _ConfiguredLlmDetector(_LoadOnDemandRuntime(llm_runtime, self._resolved_llm_model_path())),
        }
        self._factories.update(factory_overrides or {})

    def _resolved_llm_model_path(self) -> Path | None:
        if self._llm_model_path is not None:
            return self._llm_model_path
        from_environment = os.environ.get(LLM_MODEL_PATH_ENV_VAR)
        return Path(from_environment) if from_environment else None

    def names(self) -> frozenset[str]:
        return frozenset(self._factories)

    def build(self, name: str) -> Detector:
        if name not in self._factories:
            known = ", ".join(sorted(self._factories))
            raise UnknownDetectorError(f"unknown detector {name!r}; known detectors: {known}")
        return self._factories[name]()


def build_from_config(entries: Sequence[str], registry: DetectorRegistry | None = None) -> list[Detector]:
    """Builds one detector per entry; an entry `a+b` builds a union of `a` and `b`.

    Every name is checked before anything is built, so a typo fails at startup.
    """
    registry = registry or FixedDetectorRegistry()
    known = registry.names()
    for entry in entries:
        for name in entry.split(UNION_SEPARATOR):
            if name not in known:
                raise UnknownDetectorError(
                    f"unknown detector {name!r}; known detectors: {', '.join(sorted(known))}"
                )
    detectors: list[Detector] = []
    for entry in entries:
        members = tuple(registry.build(name) for name in entry.split(UNION_SEPARATOR))
        detectors.append(members[0] if len(members) == 1 else DetectorUnion(members))
    return detectors
