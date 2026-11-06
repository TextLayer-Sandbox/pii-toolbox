"""Greedy one-to-one matching of predicted spans to dataset answers, counted per common type."""

from collections.abc import Mapping, Sequence
from fractions import Fraction

from pii_toolbox.core import (
    AnswerSpan,
    MatchCounts,
    PiiType,
    Span,
    SpanMatchRule,
    SpanMatcher,
)


# A relaxed match needs the overlap to cover at least this fraction of the longer span.
MINIMUM_RELAXED_OVERLAP_FRACTION = Fraction(1, 2)


def _overlap_length(first_start: int, first_end: int, second_start: int, second_end: int) -> int:
    return min(first_end, second_end) - max(first_start, second_start)


def _group_overlapping_answers(scored_answers: Sequence[tuple[int, AnswerSpan]]) -> dict[int, int]:
    """Maps each answer index to a group id; answers of one type that overlap, directly or through a chain, share a group."""
    parent = {index: index for index, _ in scored_answers}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for position, (first_index, first) in enumerate(scored_answers):
        for second_index, second in scored_answers[position + 1 :]:
            if first.pii_type != second.pii_type:
                continue
            if _overlap_length(first.start, first.end, second.start, second.end) > 0:
                parent[find(second_index)] = find(first_index)
    return {index: find(index) for index, _ in scored_answers}


class GreedySpanMatcher(SpanMatcher):
    """Pairs a prediction with an answer of the same type; each prediction is used at most once.

    Answers of the same type that overlap, directly or through a chain of overlaps, form one
    group. A group is one entity: it counts once, is matched by at most one prediction, and
    is a hit when some member satisfies the match rule against that prediction.

    Under the relaxed rule a pair qualifies only when the overlap is at least half the length
    of the longer span. Candidate pairs are taken longest overlap first, ties broken by earlier
    start. Answers whose pii_type is None have no common counterpart and take no part in the
    counts, so a prediction that only overlaps such an answer is a false positive.
    """

    def match(
        self,
        predicted: Sequence[Span],
        answers: Sequence[AnswerSpan],
        rule: SpanMatchRule,
    ) -> Mapping[PiiType, MatchCounts]:
        scored_answers = [(index, answer) for index, answer in enumerate(answers) if answer.pii_type is not None]
        group_of_answer = _group_overlapping_answers(scored_answers)
        candidates = []
        for prediction_index, prediction in enumerate(predicted):
            for answer_index, answer in scored_answers:
                if answer.pii_type != prediction.pii_type:
                    continue
                if rule is SpanMatchRule.EXACT_BOUNDARIES and (
                    answer.start != prediction.start or answer.end != prediction.end
                ):
                    continue
                overlap = _overlap_length(prediction.start, prediction.end, answer.start, answer.end)
                if overlap <= 0:
                    continue
                longer_length = max(prediction.end - prediction.start, answer.end - answer.start)
                if (
                    rule is SpanMatchRule.ANY_OVERLAP
                    and overlap < MINIMUM_RELAXED_OVERLAP_FRACTION * longer_length
                ):
                    continue
                candidates.append((-overlap, min(prediction.start, answer.start), prediction_index, group_of_answer[answer_index]))
        candidates.sort()

        used_predictions: set[int] = set()
        used_groups: set[int] = set()
        for _, _, prediction_index, group in candidates:
            if prediction_index in used_predictions or group in used_groups:
                continue
            used_predictions.add(prediction_index)
            used_groups.add(group)

        true_positives = dict.fromkeys(PiiType, 0)
        false_positives = dict.fromkeys(PiiType, 0)
        false_negatives = dict.fromkeys(PiiType, 0)
        for prediction_index, prediction in enumerate(predicted):
            if prediction_index in used_predictions:
                true_positives[prediction.pii_type] += 1
            else:
                false_positives[prediction.pii_type] += 1
        for group, answer in {group_of_answer[index]: answer for index, answer in scored_answers}.items():
            if group not in used_groups:
                false_negatives[answer.pii_type] += 1
        return {
            pii_type: MatchCounts(true_positives[pii_type], false_positives[pii_type], false_negatives[pii_type])
            for pii_type in PiiType
        }
