import json

from pii_toolbox.core import Document, PiiType
from pii_toolbox.detection.llm_detector import PII_ITEMS_SCHEMA, LlmDetector

TEXT = "Ana Souza works at Nordwind Labs in Lisbon. Call 555-0100."


class FakeRuntime:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.schemas = []

    def generate_structured(self, prompt, json_schema):
        self.prompts.append(prompt)
        self.schemas.append(json_schema)
        return self.replies.pop(0)


def doc(text=TEXT, doc_id="d"):
    return Document(id=doc_id, text=text, language="en")


def test_valid_reply_gives_typed_spans_at_right_positions():
    reply = json.dumps(
        {
            "items": [
                {"text": "Ana Souza", "type": "PERSON"},
                {"text": "Nordwind Labs", "type": "ORGANIZATION"},
                {"text": "555-0100", "type": "IDENTIFIER"},
            ]
        }
    )
    runtime = FakeRuntime([reply])
    detector = LlmDetector(runtime)
    spans = detector.detect(doc())
    assert [(TEXT[s.start : s.end], s.pii_type) for s in spans] == [
        ("Ana Souza", PiiType.PERSON),
        ("Nordwind Labs", PiiType.ORGANIZATION),
        ("555-0100", PiiType.IDENTIFIER),
    ]
    assert all(s.source == ("llm",) for s in spans)
    assert TEXT in runtime.prompts[0]
    assert runtime.schemas[0] is PII_ITEMS_SCHEMA
    assert detector.failed_calls == 0
    assert detector.unmatched_items == 0


def test_invalid_replies_count_as_failures_and_the_loop_continues():
    good = json.dumps({"items": [{"text": "Ana Souza", "type": "PERSON"}]})
    replies = ["not json", json.dumps({"names": ["Ana"]}), "", json.dumps({"items": [{"text": 5}]}), good]
    detector = LlmDetector(FakeRuntime(replies))
    results = [detector.detect(doc(doc_id=str(i))) for i in range(5)]
    assert results[:4] == [[], [], [], []]
    assert len(results[4]) == 1
    assert detector.failed_calls == 4


def test_unmatched_items_are_counted_and_make_no_span():
    reply = json.dumps(
        {
            "items": [
                {"text": "Maria Lopez", "type": "PERSON"},
                {"text": "Lisbon", "type": "SPECIES"},
                {"text": "Lisbon", "type": "LOCATION"},
            ]
        }
    )
    detector = LlmDetector(FakeRuntime([reply]))
    spans = detector.detect(doc())
    assert [TEXT[s.start : s.end] for s in spans] == ["Lisbon"]
    assert detector.unmatched_items == 2


def test_interface_properties():
    detector = LlmDetector(FakeRuntime([]), name="llm-test")
    assert detector.name == "llm-test"
    assert detector.languages is None
    assert detector.detects == frozenset(PiiType)
    detector.load_models()
