from pii_toolbox.core import AnswerSpan, MatchCounts, PiiType, Span, SpanMatchRule
from pii_toolbox.evaluation.matcher import GreedySpanMatcher


def predicted(start, end, pii_type=PiiType.PERSON):
    return Span(start, end, pii_type, pii_type.value, ("test",), None)


def answer(start, end, pii_type=PiiType.PERSON):
    return AnswerSpan(start, end, pii_type, "LABEL", None, None, None)


def match(predictions, answers, rule=SpanMatchRule.ANY_OVERLAP):
    return GreedySpanMatcher().match(predictions, answers, rule)


def test_one_prediction_over_two_answers_is_credited_once():
    counts = match([predicted(0, 30)], [answer(0, 10), answer(20, 30)])
    assert counts[PiiType.PERSON] == MatchCounts(1, 0, 1)


def test_off_by_one_is_hit_under_any_overlap():
    counts = match([predicted(10, 19)], [answer(10, 20)], SpanMatchRule.ANY_OVERLAP)
    assert counts[PiiType.PERSON] == MatchCounts(1, 0, 0)


def test_off_by_one_is_miss_under_exact_boundaries():
    counts = match([predicted(10, 19)], [answer(10, 20)], SpanMatchRule.EXACT_BOUNDARIES)
    assert counts[PiiType.PERSON] == MatchCounts(0, 1, 1)


def test_wrong_type_is_false_positive_and_false_negative():
    counts = match([predicted(0, 5, PiiType.LOCATION)], [answer(0, 5, PiiType.PERSON)])
    assert counts[PiiType.LOCATION] == MatchCounts(0, 1, 0)
    assert counts[PiiType.PERSON] == MatchCounts(0, 0, 1)


def test_prediction_overlapping_no_answer_is_false_positive():
    counts = match([predicted(50, 60)], [answer(0, 5)])
    assert counts[PiiType.PERSON] == MatchCounts(0, 1, 1)


def test_answer_without_common_type_is_ignored():
    counts = match([], [answer(0, 5, None)])
    assert all(c == MatchCounts(0, 0, 0) for c in counts.values())


def test_prediction_over_ignored_answer_is_false_positive():
    counts = match([predicted(0, 5)], [answer(0, 5, None)])
    assert counts[PiiType.PERSON] == MatchCounts(0, 1, 0)


def test_longest_overlap_wins_the_answer():
    counts = match([predicted(0, 4), predicted(3, 12)], [answer(0, 12)])
    assert counts[PiiType.PERSON] == MatchCounts(1, 1, 0)


def test_counts_are_keyed_by_every_type():
    assert set(match([], [])) == set(PiiType)
