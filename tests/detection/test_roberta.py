import re

import pytest

from pii_toolbox.core import Document, PiiType
from pii_toolbox.detection.roberta_detector import SEGMENT_CHARACTERS, RobertaDetector
from pii_toolbox.core.model_dir import default_model_dir

# Resolved at import so the per-test sandbox home does not hide the downloaded models.
MODEL_DIR = default_model_dir()

TEXT = "Maria Gonzalez works at Acme Corp in Lisbon."
NAME = "Maria Gonzalez"


def stub_entity(label, value, text=TEXT):
    start = text.index(value)
    return {"entity_group": label, "score": 0.99, "start": start, "end": start + len(value)}


def test_misc_is_dropped_and_other_labels_are_typed():
    entities = [
        stub_entity("PER", NAME),
        stub_entity("ORG", "Acme Corp"),
        stub_entity("LOC", "Lisbon"),
        stub_entity("MISC", "works"),
    ]
    detector = RobertaDetector(pipeline=lambda text: entities)
    spans = detector.detect(Document(id="d", text=TEXT, language="en"))
    assert [(TEXT[s.start : s.end], s.pii_type) for s in spans] == [
        (NAME, PiiType.PERSON),
        ("Acme Corp", PiiType.ORGANIZATION),
        ("Lisbon", PiiType.LOCATION),
    ]
    assert all(s.source == ("roberta",) and s.score == 0.99 for s in spans)


def test_interface_properties():
    detector = RobertaDetector(pipeline=lambda text: [])
    assert detector.name == "roberta"
    assert detector.languages == frozenset({"en"})
    assert detector.detects == frozenset({PiiType.PERSON, PiiType.ORGANIZATION, PiiType.LOCATION})


def test_long_text_gives_spans_at_correct_positions_across_segment_joins():
    # A name straddling the first segment's hard cut, then names every 5,000 characters, then one at the end.
    head = "x" * (SEGMENT_CHARACTERS - 5) + NAME
    body = ""
    while len(head) + len(body) + 5000 < 99_000:
        body += "y" * (5000 - len(NAME) - 1) + " " + NAME
    text = head + body + "z" * (100_000 - len(head) - len(body) - len(NAME)) + NAME
    assert len(text) == 100_000

    def stub_pipeline(segment):
        if len(segment) > SEGMENT_CHARACTERS:
            return []
        return [
            {"entity_group": "PER", "score": 0.9, "start": m.start(), "end": m.end()}
            for m in re.finditer(NAME, segment)
        ]

    detector = RobertaDetector(pipeline=stub_pipeline)
    spans = detector.detect(Document(id="d", text=text, language="en"))
    expected = [m.start() for m in re.finditer(NAME, text)]
    assert expected[0] == SEGMENT_CHARACTERS - 5
    assert expected[-1] == 100_000 - len(NAME)
    assert [s.start for s in spans] == expected
    assert all(text[s.start : s.end] == NAME for s in spans)


@pytest.mark.model
def test_real_model_finds_the_person():
    detector = RobertaDetector(MODEL_DIR)
    detector.load_models()
    spans = detector.detect(Document(id="d", text=TEXT, language="en"))
    assert any(s.pii_type is PiiType.PERSON and TEXT[s.start : s.end] == NAME for s in spans)


def test_punctuation_only_span_is_dropped_and_name_remains():
    text = "Maria Gonzalez works at Acme Corp."
    dot = text.rindex(".")
    entities = [
        {"entity_group": "PER", "score": 0.9, "start": 0, "end": len(NAME)},
        {"entity_group": "ORG", "score": 0.9, "start": dot, "end": dot + 1},
    ]
    spans = RobertaDetector(pipeline=lambda segment: entities).detect(Document(id="d", text=text, language="en"))
    assert [text[s.start : s.end] for s in spans] == [NAME]
