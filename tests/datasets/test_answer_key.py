from pii_toolbox.core import AnswerSpan, PiiType, ReidentificationRisk
from pii_toolbox.datasets.answer_key import AnyAnnotatorAnswerKeyBuilder

DIRECT = ReidentificationRisk.DIRECT
QUASI = ReidentificationRisk.QUASI
NO_MASK = ReidentificationRisk.NONE


def mark(annotator, start, end, risk, label="PERSON", pii_type=PiiType.PERSON):
    return AnswerSpan(start, end, pii_type, label, annotator, risk, f"{annotator}-e")


def build(answers):
    return AnyAnnotatorAnswerKeyBuilder().build(answers)


def bounds(key):
    return [(span.start, span.end) for span in key]


def test_distinct_boundaries_stay_distinct():
    key = build([mark("A", 0, 12, DIRECT), mark("B", 0, 8, QUASI)])
    assert bounds(key) == [(0, 8), (0, 12)]


def test_direct_by_one_and_no_mask_by_another_is_in_the_key():
    key = build([mark("A", 0, 12, DIRECT), mark("B", 0, 12, NO_MASK)])
    assert bounds(key) == [(0, 12)]
    assert key[0].risk is DIRECT


def test_span_marked_only_no_mask_is_not_in_the_key():
    assert build([mark("A", 0, 5, NO_MASK), mark("B", 0, 5, NO_MASK)]) == ()


def test_dem_with_no_mask_is_not_in_the_key():
    dem = mark("A", 0, 5, NO_MASK, label="DEM", pii_type=None)
    assert build([dem]) == ()


def test_identical_span_from_two_annotators_counts_once_with_stronger_risk():
    key = build([mark("A", 0, 12, QUASI), mark("B", 0, 12, DIRECT)])
    assert len(key) == 1
    assert key[0].risk is DIRECT
    assert key[0].annotator is None


def test_single_annotator_span_keeps_its_annotator():
    key = build([mark("A", 0, 12, DIRECT)])
    assert key[0].annotator == "A"


def test_answers_without_risk_grading_are_kept():
    assert bounds(build([mark(None, 3, 9, None)])) == [(3, 9)]
