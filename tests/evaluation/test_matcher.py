from pii_toolbox.core import AnswerSpan, MatchCounts, PiiType, Span, SpanMatchRule
from pii_toolbox.evaluation.matcher import GreedySpanMatcher


def predicted(start, end, pii_type=PiiType.PERSON):
    return Span(start, end, pii_type, pii_type.value, ("test",), None)


def answer(start, end, pii_type: PiiType | None = PiiType.PERSON):
    return AnswerSpan(start, end, pii_type, "LABEL", None, None, None)


def match(predictions, answers, rule=SpanMatchRule.ANY_OVERLAP):
    return GreedySpanMatcher().match(predictions, answers, rule)


def test_one_prediction_over_two_answers_is_credited_once():
    counts = match([predicted(0, 20)], [answer(0, 10), answer(10, 20)])
    assert counts[PiiType.PERSON] == MatchCounts(1, 0, 1)


def test_overlap_of_exactly_half_the_longer_span_is_hit_under_any_overlap():
    counts = match([predicted(0, 10)], [answer(5, 15)])
    assert counts[PiiType.PERSON] == MatchCounts(1, 0, 0)


def test_overlap_just_under_half_the_longer_span_is_miss_under_any_overlap():
    counts = match([predicted(0, 10)], [answer(6, 16)])
    assert counts[PiiType.PERSON] == MatchCounts(0, 1, 1)


def test_missing_title_is_hit_under_any_overlap():
    text = "Mr Galip Yalman"
    start = text.index("Galip")
    counts = match([predicted(start, len(text))], [answer(0, len(text))])
    assert counts[PiiType.PERSON] == MatchCounts(1, 0, 0)


def test_one_character_overlap_is_miss_under_any_overlap():
    counts = match([predicted(0, 10)], [answer(9, 19)])
    assert counts[PiiType.PERSON] == MatchCounts(0, 1, 1)


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


def test_overlapping_same_type_answers_are_one_hit_when_a_prediction_matches_either():
    for prediction in (predicted(0, 10), predicted(5, 15)):
        counts = match([prediction], [answer(0, 10), answer(5, 15)])
        assert counts[PiiType.PERSON] == MatchCounts(1, 0, 0)


def test_overlapping_same_type_answers_are_one_miss_without_a_prediction():
    counts = match([], [answer(0, 10), answer(5, 15)])
    assert counts[PiiType.PERSON] == MatchCounts(0, 0, 1)


def test_chained_overlapping_answers_are_one_group():
    counts = match([predicted(0, 10)], [answer(0, 10), answer(8, 18), answer(16, 26)])
    assert counts[PiiType.PERSON] == MatchCounts(1, 0, 0)
    assert match([], [answer(0, 10), answer(8, 18), answer(16, 26)])[PiiType.PERSON] == MatchCounts(0, 0, 1)


def test_touching_answers_are_not_grouped():
    counts = match([], [answer(0, 10), answer(10, 20)])
    assert counts[PiiType.PERSON] == MatchCounts(0, 0, 2)


def test_overlapping_answers_of_different_types_stay_separate():
    answers = [answer(0, 10), answer(5, 15, PiiType.ORGANIZATION)]
    counts = match([predicted(0, 10)], answers)
    assert counts[PiiType.PERSON] == MatchCounts(1, 0, 0)
    assert counts[PiiType.ORGANIZATION] == MatchCounts(0, 0, 1)


def test_strict_hit_on_either_group_member_exact_offsets():
    for prediction in (predicted(0, 10), predicted(5, 15)):
        counts = match([prediction], [answer(0, 10), answer(5, 15)], SpanMatchRule.EXACT_BOUNDARIES)
        assert counts[PiiType.PERSON] == MatchCounts(1, 0, 0)


def test_strict_prediction_matching_no_member_exactly_is_miss_and_false_positive():
    counts = match([predicted(0, 12)], [answer(0, 10), answer(5, 15)], SpanMatchRule.EXACT_BOUNDARIES)
    assert counts[PiiType.PERSON] == MatchCounts(0, 1, 1)
