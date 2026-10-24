"""Greedy one-to-one matching of predicted spans to dataset answers, counted per common type."""

from collections.abc import Mapping, Sequence

from pii_toolbox.core import (
    AnswerSpan,
    MatchCounts,
    PiiType,
    Span,
    SpanMatchRule,
    SpanMatcher,
)


def _overlap_length(first_start: int, first_end: int, second_start: int, second_end: int) -> int:
    return min(first_end, second_end) - max(first_start, second_start)


class GreedySpanMatcher(SpanMatcher):
    """Pairs a prediction with an answer of the same type; each is used at most once.

    Candidate pairs are taken longest overlap first, ties broken by earlier start. Answers
    whose pii_type is None have no common counterpart and take no part in the counts, so a
    prediction that only overlaps such an answer is a false positive.
    """

    def match(
        self,
        predicted: Sequence[Span],
        answers: Sequence[AnswerSpan],
        rule: SpanMatchRule,
    ) -> Mapping[PiiType, MatchCounts]:
        scored_answers = [(index, answer) for index, answer in enumerate(answers) if answer.pii_type is not None]
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
                candidates.append((-overlap, min(prediction.start, answer.start), prediction_index, answer_index))
        candidates.sort()

        used_predictions: set[int] = set()
        used_answers: set[int] = set()
        for _, _, prediction_index, answer_index in candidates:
            if prediction_index in used_predictions or answer_index in used_answers:
                continue
            used_predictions.add(prediction_index)
            used_answers.add(answer_index)

        true_positives = dict.fromkeys(PiiType, 0)
        false_positives = dict.fromkeys(PiiType, 0)
        false_negatives = dict.fromkeys(PiiType, 0)
        for prediction_index, prediction in enumerate(predicted):
            if prediction_index in used_predictions:
                true_positives[prediction.pii_type] += 1
            else:
                false_positives[prediction.pii_type] += 1
        for answer_index, answer in scored_answers:
            if answer_index not in used_answers:
                false_negatives[answer.pii_type] += 1
        return {
            pii_type: MatchCounts(true_positives[pii_type], false_positives[pii_type], false_negatives[pii_type])
            for pii_type in PiiType
        }
