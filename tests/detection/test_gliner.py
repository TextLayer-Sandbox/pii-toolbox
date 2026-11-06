import re

import pytest

from pii_toolbox.core import Document, PiiType
from pii_toolbox.detection.gliner_detector import (
    REQUESTED_LABELS,
    SEGMENT_CHARACTERS,
    GlinerDetector,
)
from pii_toolbox.evaluation.model_manifest import default_model_dir

# Resolved at import so the per-test sandbox home does not hide the downloaded models.
MODEL_DIR = default_model_dir()

TEXT = (
    "Maria Gonzalez works at Acme Corp in Lisbon, mail maria@example.com. "
    "On 3 May 2001 she paid EUR 2,500 in case 36110/97 under ID 1234567890."
)
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
            entity(text, "date", "3 May 2001"),
            entity(text, "money", "EUR 2,500"),
            entity(text, "case number", "36110/97"),
            entity(text, "identification number", "1234567890"),
        ]

    model = StubModel(entities_for)
    spans = GlinerDetector(model=model).detect(Document(id="d", text=TEXT, language="en"))
    assert [(TEXT[s.start : s.end], s.pii_type) for s in spans] == [
        (NAME, PiiType.PERSON),
        ("Acme Corp", PiiType.ORGANIZATION),
        ("Lisbon", PiiType.LOCATION),
        ("3 May 2001", PiiType.DATE_TIME),
        ("EUR 2,500", PiiType.QUANTITY),
        ("36110/97", PiiType.IDENTIFIER),
        ("1234567890", PiiType.IDENTIFIER),
    ]
    assert not any(s.detector_label == "email" for s in spans)
    assert model.asked[0] == list(REQUESTED_LABELS)


def test_requested_labels_and_interface_properties():
    assert REQUESTED_LABELS == (
        "person",
        "organization",
        "location",
        "address",
        "date",
        "money",
        "case number",
        "identification number",
    )
    detector = GlinerDetector(model=StubModel(lambda text: []))
    assert detector.name == "gliner"
    assert detector.languages is None
    assert detector.detects == frozenset(
        {
            PiiType.PERSON,
            PiiType.ORGANIZATION,
            PiiType.LOCATION,
            PiiType.DATE_TIME,
            PiiType.QUANTITY,
            PiiType.IDENTIFIER,
        }
    )


def test_long_text_gives_spans_at_correct_positions_across_segment_joins():
    # A name straddling the first segment's hard cut, then names every 5,000 characters, then one at the end.
    head = "x" * (SEGMENT_CHARACTERS - 5) + NAME
    body = ""
    while len(head) + len(body) + 5000 < 99_000:
        body += "y" * (5000 - len(NAME) - 1) + " " + NAME
    text = head + body + "z" * (100_000 - len(head) - len(body) - len(NAME)) + NAME
    assert len(text) == 100_000

    def entities_for(segment):
        assert len(segment) <= SEGMENT_CHARACTERS, "model was sent more than one segment of text"
        return [
            {"start": m.start(), "end": m.end(), "label": "person", "score": 0.9}
            for m in re.finditer(NAME, segment)
        ]

    spans = GlinerDetector(model=StubModel(entities_for)).detect(Document(id="d", text=text, language="en"))
    expected = [m.start() for m in re.finditer(NAME, text)]
    assert expected[0] == SEGMENT_CHARACTERS - 5
    assert expected[-1] == 100_000 - len(NAME)
    assert [s.start for s in spans] == expected
    assert all(text[s.start : s.end] == NAME for s in spans)


@pytest.mark.model
def test_real_model_finds_the_person():
    detector = GlinerDetector(MODEL_DIR)
    detector.load_models()
    spans = detector.detect(Document(id="d", text=TEXT, language="en"))
    assert any(s.pii_type is PiiType.PERSON and TEXT[s.start : s.end] == NAME for s in spans)


@pytest.mark.model
def test_real_model_finds_names_at_correct_positions_in_long_text():
    sentence = "The weather was calm and the market was quiet today. "
    pieces = []
    length = 0
    next_name = 5000
    while length < 100_000:
        if length >= next_name:
            pieces.append(f"{NAME} works at Acme Corp. ")
            next_name += 5000
        else:
            pieces.append(sentence)
        length += len(pieces[-1])
    text = "".join(pieces)
    detector = GlinerDetector(MODEL_DIR)
    detector.load_models()
    spans = detector.detect(Document(id="d", text=text, language="en"))
    expected = {m.start() for m in re.finditer(NAME, text)}
    found = {s.start for s in spans if s.pii_type is PiiType.PERSON and text[s.start : s.end] == NAME}
    assert found <= expected
    assert len(found) >= 0.8 * len(expected)


def test_punctuation_only_span_is_dropped_and_name_remains():
    text = "Maria Gonzalez works at Acme Corp."
    dot = text.rindex(".")
    entities = [
        {"start": 0, "end": len(NAME), "label": "person", "score": 0.9},
        {"start": dot, "end": dot + 1, "label": "organization", "score": 0.9},
    ]
    detector = GlinerDetector(model=StubModel(lambda segment: entities))
    spans = detector.detect(Document(id="d", text=text, language="en"))
    assert [text[s.start : s.end] for s in spans] == [NAME]
