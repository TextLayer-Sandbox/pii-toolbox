import re

import pytest

from pii_toolbox.core import Document, PiiType
from pii_toolbox.detection.gliner_detector import REQUESTED_LABELS, GlinerDetector
from pii_toolbox.evaluation.model_manifest import default_model_dir

# Resolved at import so the per-test sandbox home does not hide the downloaded models.
MODEL_DIR = default_model_dir()

TEXT = "Maria Gonzalez works at Acme Corp in Lisbon, mail maria@example.com."
NAME = "Maria Gonzalez"


class StubModel:
    def __init__(self, entities_for):
        self.entities_for = entities_for
        self.asked = []

    def predict_entities(self, text, labels, threshold):
        self.asked.append(list(labels))
        return self.entities_for(text)


def entity(text, label, value):
    start = text.index(value)
    return {"start": start, "end": start + len(value), "label": label, "score": 0.9, "text": value}


def test_labels_map_to_common_types_and_unlisted_labels_are_dropped():
    def entities_for(text):
        return [
            entity(text, "person", NAME),
            entity(text, "organization", "Acme Corp"),
            entity(text, "location", "Lisbon"),
            entity(text, "address", "Lisbon"),
            entity(text, "email", "maria@example.com"),
        ]

    model = StubModel(entities_for)
    spans = GlinerDetector(model=model).detect(Document(id="d", text=TEXT, language="en"))
    assert [(TEXT[s.start : s.end], s.pii_type) for s in spans] == [
        (NAME, PiiType.PERSON),
        ("Acme Corp", PiiType.ORGANIZATION),
        ("Lisbon", PiiType.LOCATION),
    ]
    assert not any(s.detector_label == "email" for s in spans)
    assert model.asked[0] == ["person", "organization", "location", "address"]


def test_requested_labels_and_interface_properties():
    assert REQUESTED_LABELS == ("person", "organization", "location", "address")
    detector = GlinerDetector(model=StubModel(lambda text: []))
    assert detector.name == "gliner"
    assert detector.languages is None
    assert detector.detects == frozenset({PiiType.PERSON, PiiType.ORGANIZATION, PiiType.LOCATION})


def test_long_text_gives_spans_at_correct_positions_across_segment_joins():
    filler = "lorem ipsum dolor sit amet "
    block = filler * ((5000 - len(NAME) - 1) // len(filler)) + NAME + " "
    text = (block * 21)[:100_000]

    def entities_for(segment):
        return [
            {"start": m.start(), "end": m.end(), "label": "person", "score": 0.9}
            for m in re.finditer(NAME, segment)
        ]

    spans = GlinerDetector(model=StubModel(entities_for)).detect(Document(id="d", text=text, language="en"))
    expected = [m.start() for m in re.finditer(NAME, text)]
    assert len(expected) >= 19
    assert [s.start for s in spans] == expected
    assert all(text[s.start : s.end] == NAME for s in spans)


@pytest.mark.model
def test_real_model_finds_the_person():
    detector = GlinerDetector(MODEL_DIR)
    detector.load_models()
    spans = detector.detect(Document(id="d", text=TEXT, language="en"))
    assert any(s.pii_type is PiiType.PERSON and TEXT[s.start : s.end] == NAME for s in spans)
