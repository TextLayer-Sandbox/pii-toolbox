import dataclasses

import pytest

from pii_toolbox import core
from pii_toolbox.core import Detector, Document, PiiType, Span

SHAPE_NAMES = [
    "Span", "Document", "PiiType", "LabelMap", "ReidentificationRisk", "AnswerSpan",
    "AnnotatedDocument", "EntityGroup", "Detector", "DetectorUnion", "DetectorRegistry",
    "TextSegment", "TextSegmenter", "IdentityResolver", "Placeholder", "PlaceholderToken",
    "PlaceholderFormat", "MaskResult", "RestoreResult", "PlaceholderStore",
    "RedactionSession", "DatasetSplit", "Dataset", "SpanMatchRule", "MatchCounts",
    "DetectionScore", "DetectionReport", "SpanMatcher", "DetectionScorer",
    "AnswerKeyBuilder", "TabOfficialMetrics", "TabMetricsScorer", "CostMeasurement",
    "CostRunner", "RunEnvironment", "EvaluationRun",
]


def make_span(start: int, end: int) -> Span:
    return Span(start, end, PiiType.PERSON, "PERSON", ("test",), None)


@pytest.mark.parametrize("name", SHAPE_NAMES)
def test_shared_name_is_exported(name):
    assert hasattr(core, name)


def test_span_is_frozen():
    span = make_span(0, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.start = 1  # type: ignore[misc]


@pytest.mark.parametrize("start, end", [(5, 5), (-1, 3), (6, 5)])
def test_span_rejects_invalid_bounds(start, end):
    with pytest.raises(ValueError):
        make_span(start, end)


def test_detector_without_detect_cannot_be_instantiated():
    class Incomplete(Detector):
        name = "incomplete"
        languages = None
        detects = frozenset()

        def load_models(self) -> None:
            return None

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_pii_type_has_exactly_six_members():
    assert {member.name for member in PiiType} == {
        "PERSON", "ORGANIZATION", "LOCATION", "DATE_TIME", "QUANTITY", "IDENTIFIER",
    }
    assert len(PiiType) == 6


def test_document_holds_its_fields():
    document = Document(id="d1", text="Alice Example", language="en")
    assert document.text == "Alice Example"
