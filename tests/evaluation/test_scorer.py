import pytest

from pii_toolbox.core import AnswerSpan, Document, PiiType, Span, SpanMatchRule
from pii_toolbox.evaluation.scorer import MatchingDetectionScorer


def predicted(start, end, pii_type):
    return Span(start, end, pii_type, pii_type.value, ("test",), None)


def answer(start, end, pii_type):
    return AnswerSpan(start, end, pii_type, "LABEL", None, None, None)


def unbalanced_fixture(language="en"):
    """PERSON tp=8 fp=2 fn=2; LOCATION tp=1 fp=0 fn=3."""
    person = PiiType.PERSON
    location = PiiType.LOCATION
    answers = [answer(i * 10, i * 10 + 5, person) for i in range(10)]
    answers += [answer(1000 + i * 10, 1000 + i * 10 + 5, location) for i in range(4)]
    spans = [predicted(i * 10, i * 10 + 5, person) for i in range(8)]
    spans += [predicted(500 + i * 10, 505 + i * 10, person) for i in range(2)]
    spans += [predicted(1000, 1005, location)]
    return [(Document("d", "x" * 2000, language), spans, answers)]


def f2(precision, recall):
    return 5 * precision * recall / (4 * precision + recall)


def score(results, rule=SpanMatchRule.ANY_OVERLAP, **kwargs):
    return MatchingDetectionScorer().score(results, rule, **kwargs)


def test_per_type_scores():
    report = score(unbalanced_fixture())
    person = report.by_type[PiiType.PERSON]
    assert (person.precision, person.recall, person.f2) == pytest.approx((0.8, 0.8, 0.8))
    location = report.by_type[PiiType.LOCATION]
    assert location.precision == 1.0
    assert location.recall == 0.25
    assert location.f2 == pytest.approx(5 * 1.0 * 0.25 / (4 * 1.0 + 0.25))


def test_overall_micro_pools_counts():
    micro = score(unbalanced_fixture()).overall_micro
    assert micro.precision == pytest.approx(9 / 11)
    assert micro.recall == pytest.approx(9 / 14)
    assert micro.f2 == pytest.approx(f2(9 / 11, 9 / 14))


def test_macro_is_plain_mean_of_per_type_scores_over_reported_types():
    report = score(unbalanced_fixture(), reported_types=frozenset({PiiType.PERSON, PiiType.LOCATION}))
    location_f2 = 5 * 0.25 / 4.25
    assert report.macro.precision == pytest.approx((0.8 + 1.0) / 2)
    assert report.macro.recall == pytest.approx((0.8 + 0.25) / 2)
    assert report.macro.f2 == pytest.approx((0.8 + location_f2) / 2)
    assert report.macro.f2 != pytest.approx(report.overall_micro.f2)


def test_unreported_type_is_none_and_excluded_from_macro():
    report = score(unbalanced_fixture(), reported_types=frozenset({PiiType.PERSON}))
    assert report.by_type[PiiType.LOCATION] is None
    assert report.macro.f2 == pytest.approx(0.8)
    assert report.overall_micro.recall == pytest.approx(9 / 14)


def test_empty_input_gives_zeros_not_an_error():
    report = score([])
    assert report.overall_micro.f2 == 0.0
    assert report.macro.f2 == 0.0
    assert report.by_type[PiiType.PERSON].precision == 0.0
    assert report.by_language == {}


def test_by_language_uses_document_language():
    results = unbalanced_fixture("en") + unbalanced_fixture("fr")[:1]
    report = score(results)
    assert set(report.by_language) == {"en", "fr"}
    assert report.by_language["en"].recall == pytest.approx(9 / 14)


def test_each_rule_has_its_own_report():
    spans = [predicted(10, 19, PiiType.PERSON)]
    results = [(Document("d", "x" * 100, "en"), spans, [answer(10, 20, PiiType.PERSON)])]
    relaxed = score(results, SpanMatchRule.ANY_OVERLAP)
    strict = score(results, SpanMatchRule.EXACT_BOUNDARIES)
    assert relaxed.rule is SpanMatchRule.ANY_OVERLAP
    assert strict.rule is SpanMatchRule.EXACT_BOUNDARIES
    assert relaxed.by_type[PiiType.PERSON].f2 == 1.0
    assert strict.by_type[PiiType.PERSON].f2 == 0.0
