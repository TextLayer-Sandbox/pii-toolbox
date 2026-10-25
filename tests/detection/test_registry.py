import json
import sys
import types
from types import SimpleNamespace

import pytest

from pii_toolbox.core import Document, PiiType
from pii_toolbox.detection.gliner_detector import GlinerDetector
from pii_toolbox.detection.llm_detector import LlmDetector
from pii_toolbox.detection.registry import (
    FixedDetectorRegistry,
    UnknownDetectorError,
    build_from_config,
)
from pii_toolbox.detection.roberta_detector import RobertaDetector
from pii_toolbox.detection.spacy_detectors import SpacyDetector
from pii_toolbox.detection.union import DetectorUnion
from pii_toolbox.evaluation.model_manifest import default_model_dir

# Resolved at import so the per-test sandbox home does not hide the downloaded models.
MODEL_DIR = default_model_dir()

TEXT = "Maria Gonzalez works at Acme Corp in Lisbon."
DOCUMENT = Document(id="d", text=TEXT, language="en")
NAME = "Maria Gonzalez"
EXPECTED_NAMES = {"presidio_structured", "spacy_lg", "spacy_sm", "roberta", "gliner", "llm"}


def stub_registry():
    start = TEXT.index(NAME)

    def spacy_stub(name):
        entity = SimpleNamespace(start_char=start, end_char=start + len(NAME), label_="PERSON")
        return lambda: SpacyDetector(name, "stub", pipeline=lambda text: SimpleNamespace(ents=[entity]))

    class LlmRuntimeStub:
        def generate_structured(self, prompt, json_schema):
            return json.dumps({"items": [{"text": NAME, "type": "PERSON"}]})

    return FixedDetectorRegistry(
        factory_overrides={
            "spacy_lg": spacy_stub("spacy_lg"),
            "spacy_sm": spacy_stub("spacy_sm"),
            "roberta": lambda: RobertaDetector(
                pipeline=lambda text: [{"entity_group": "PER", "score": 0.9, "start": start, "end": start + len(NAME)}]
            ),
            "gliner": lambda: GlinerDetector(
                model=SimpleNamespace(
                    predict_entities=lambda text, labels, threshold: [
                        {"start": start, "end": start + len(NAME), "label": "person", "score": 0.9}
                    ]
                )
            ),
            "llm": lambda: LlmDetector(LlmRuntimeStub()),
        }
    )


def test_registry_names():
    assert FixedDetectorRegistry().names() == EXPECTED_NAMES


def test_unknown_name_raises_at_startup_listing_every_known_name():
    with pytest.raises(UnknownDetectorError) as error:
        build_from_config(["nonexistent"])
    for name in EXPECTED_NAMES:
        assert name in str(error.value)
    assert "nonexistent" in str(error.value)


def test_unknown_member_of_a_union_is_rejected_before_anything_is_built():
    built = []
    registry = FixedDetectorRegistry(factory_overrides={"spacy_sm": lambda: built.append("x")})
    with pytest.raises(UnknownDetectorError):
        build_from_config(["spacy_sm+nonexistent"], registry)
    assert built == []


def test_presidio_structured_alone_builds_only_that_detector():
    detectors = build_from_config(["presidio_structured"])
    assert [detector.name for detector in detectors] == ["presidio_structured"]


def test_plus_builds_a_union():
    detectors = build_from_config(["spacy_lg+roberta", "gliner"], stub_registry())
    assert isinstance(detectors[0], DetectorUnion)
    assert [member.name for member in detectors[0].members] == ["spacy_lg", "roberta"]
    assert detectors[1].name == "gliner"


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES - {"presidio_structured"}))
def test_stubbed_detectors_return_valid_spans(name):
    spans = stub_registry().build(name).detect(DOCUMENT)
    assert spans
    for span in spans:
        assert 0 <= span.start < span.end <= len(TEXT)


def test_properties_of_each_detector():
    registry = stub_registry()
    for name in ("spacy_lg", "spacy_sm", "roberta"):
        assert registry.build(name).languages == frozenset({"en"})
    assert registry.build("gliner").languages is None
    assert FixedDetectorRegistry().build("presidio_structured").detects == frozenset({PiiType.IDENTIFIER})


def test_llm_builds_without_a_model_but_cannot_load(monkeypatch):
    monkeypatch.delenv("PII_LLM_MODEL_PATH", raising=False)
    detector = FixedDetectorRegistry().build("llm")
    with pytest.raises(FileNotFoundError, match="PII_LLM_MODEL_PATH"):
        detector.load_models()


def test_llm_model_path_comes_from_the_environment(monkeypatch, tmp_path):
    missing = tmp_path / "absent.gguf"
    monkeypatch.setenv("PII_LLM_MODEL_PATH", str(missing))
    monkeypatch.setitem(sys.modules, "llama_cpp", types.ModuleType("llama_cpp"))
    with pytest.raises(FileNotFoundError, match="absent.gguf"):
        FixedDetectorRegistry().build("llm").load_models()


@pytest.mark.model
@pytest.mark.parametrize("name", ["spacy_lg", "spacy_sm", "roberta", "gliner"])
def test_real_models_built_through_the_registry_find_the_person(name):
    detector = FixedDetectorRegistry(model_dir=MODEL_DIR).build(name)
    detector.load_models()
    spans = detector.detect(DOCUMENT)
    assert any(s.pii_type is PiiType.PERSON and TEXT[s.start : s.end] == NAME for s in spans)
