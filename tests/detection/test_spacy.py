from types import SimpleNamespace

import pytest

from pii_toolbox.core import Document, PiiType
from pii_toolbox.detection.spacy_detectors import (
    SpacyDetector,
    spacy_lg_detector,
    spacy_sm_detector,
)
from pii_toolbox.evaluation.model_manifest import default_model_dir

# Resolved at import so the per-test sandbox home does not hide the downloaded models.
MODEL_DIR = default_model_dir()

TEXT = "Maria Gonzalez works at Acme Corp in Lisbon."
DOCUMENT = Document(id="d", text=TEXT, language="en")


def entity(text, label):
    start = TEXT.index(text) if text in TEXT else 0
    return SimpleNamespace(start_char=start, end_char=start + len(text), label_=label)


def test_spacy_labels_map_to_common_types_with_stub_pipeline():
    entities = [
        entity("Maria Gonzalez", "PERSON"),
        entity("Acme Corp", "ORG"),
        entity("Lisbon", "GPE"),
        entity("works", "CARDINAL"),
    ]
    detector = SpacyDetector("stub", "stub_model", pipeline=lambda text: SimpleNamespace(ents=entities))
    spans = detector.detect(DOCUMENT)
    assert [(TEXT[s.start : s.end], s.pii_type, s.detector_label) for s in spans] == [
        ("Maria Gonzalez", PiiType.PERSON, "PERSON"),
        ("Acme Corp", PiiType.ORGANIZATION, "ORG"),
        ("Lisbon", PiiType.LOCATION, "GPE"),
    ]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("LOC", PiiType.LOCATION),
        ("DATE", PiiType.DATE_TIME),
        ("TIME", PiiType.DATE_TIME),
        ("MONEY", PiiType.QUANTITY),
        ("QUANTITY", PiiType.QUANTITY),
    ],
)
def test_other_labels_map_to_common_types(label, expected):
    detector = SpacyDetector("stub", "m", pipeline=lambda text: SimpleNamespace(ents=[entity("Lisbon", label)]))
    assert detector.detect(DOCUMENT)[0].pii_type is expected


def test_interface_properties():
    detector = spacy_sm_detector()
    assert detector.name == "spacy_sm"
    assert spacy_lg_detector().name == "spacy_lg"
    assert detector.languages == frozenset({"en"})
    assert {
        PiiType.PERSON,
        PiiType.ORGANIZATION,
        PiiType.LOCATION,
        PiiType.DATE_TIME,
        PiiType.QUANTITY,
    } <= detector.detects


@pytest.mark.model
@pytest.mark.parametrize("factory", [spacy_lg_detector, spacy_sm_detector])
def test_real_model_finds_the_person(factory):
    detector = factory(MODEL_DIR)
    detector.load_models()
    spans = detector.detect(DOCUMENT)
    assert any(s.pii_type is PiiType.PERSON and TEXT[s.start : s.end] == "Maria Gonzalez" for s in spans)
    assert all(0 <= s.start < s.end <= len(TEXT) for s in spans)


def test_punctuation_only_span_is_dropped_and_name_remains():
    text = "Maria Gonzalez works at Acme Corp."
    dot = text.rindex(".")
    entities = [
        SimpleNamespace(start_char=0, end_char=len("Maria Gonzalez"), label_="PERSON"),
        SimpleNamespace(start_char=dot, end_char=dot + 1, label_="ORG"),
    ]
    detector = SpacyDetector("stub", "stub_model", pipeline=lambda t: SimpleNamespace(ents=entities))
    spans = detector.detect(Document(id="d", text=text, language="en"))
    assert [text[s.start : s.end] for s in spans] == ["Maria Gonzalez"]
