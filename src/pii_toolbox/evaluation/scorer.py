"""Precision, recall and F2 over matched spans, per type, pooled, averaged and per language."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from pii_toolbox.core import (
    AnswerSpan,
    DetectionReport,
    DetectionScore,
    DetectionScorer,
    Document,
    MatchCounts,
    PiiType,
    Span,
    SpanMatchRule,
)
from pii_toolbox.evaluation.matcher import GreedySpanMatcher

F_BETA = 2


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def score_from_counts(counts: MatchCounts) -> DetectionScore:
    precision = _ratio(counts.true_positives, counts.true_positives + counts.false_positives)
    recall = _ratio(counts.true_positives, counts.true_positives + counts.false_negatives)
    return _score(precision, recall)


def _score(precision: float, recall: float) -> DetectionScore:
    beta_squared = F_BETA**2
    f2 = _ratio((1 + beta_squared) * precision * recall, beta_squared * precision + recall)
    return DetectionScore(precision, recall, f2)


def _add(first: MatchCounts, second: MatchCounts) -> MatchCounts:
    return MatchCounts(
        first.true_positives + second.true_positives,
        first.false_positives + second.false_positives,
        first.false_negatives + second.false_negatives,
    )


def _pool(counts: Iterable[MatchCounts]) -> MatchCounts:
    pooled = MatchCounts(0, 0, 0)
    for item in counts:
        pooled = _add(pooled, item)
    return pooled


class MatchingDetectionScorer(DetectionScorer):
    """Scores detector output against answers using greedy one-to-one matching.

    Beyond the abstract signature, `score` takes an optional keyword `reported_types`
    (default: all six types): the types the detector can report. An unreported type is
    None in the per-type breakdown and is left out of the macro average; its answers still
    count as misses in the pooled overall score.
    """

    def score(
        self,
        results: Iterable[tuple[Document, Sequence[Span], Sequence[AnswerSpan]]],
        rule: SpanMatchRule,
        *,
        reported_types: frozenset[PiiType] = frozenset(PiiType),
    ) -> DetectionReport:
        matcher = GreedySpanMatcher()
        per_type: dict[PiiType, MatchCounts] = dict.fromkeys(PiiType, MatchCounts(0, 0, 0))
        per_language: dict[str, MatchCounts] = defaultdict(lambda: MatchCounts(0, 0, 0))
        for document, predicted, answers in results:
            counts = matcher.match(predicted, answers, rule)
            for pii_type, type_counts in counts.items():
                per_type[pii_type] = _add(per_type[pii_type], type_counts)
            per_language[document.language] = _add(per_language[document.language], _pool(counts.values()))

        by_type: Mapping[PiiType, DetectionScore | None] = {
            pii_type: score_from_counts(per_type[pii_type]) if pii_type in reported_types else None
            for pii_type in PiiType
        }
        reported_scores = [score for score in by_type.values() if score is not None]
        count = len(reported_scores)
        macro = DetectionScore(
            _ratio(sum(score.precision for score in reported_scores), count),
            _ratio(sum(score.recall for score in reported_scores), count),
            _ratio(sum(score.f2 for score in reported_scores), count),
        )
        return DetectionReport(
            rule=rule,
            overall_micro=score_from_counts(_pool(per_type.values())),
            macro=macro,
            by_type=by_type,
            by_language={language: score_from_counts(counts) for language, counts in per_language.items()},
        )
