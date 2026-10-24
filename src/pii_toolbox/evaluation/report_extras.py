"""Scores that sit beside the shared detection report: TAB's extra label rows and the overall that includes IDENTIFIER."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from pii_toolbox.core import (
    AnswerSpan,
    DetectionReport,
    DetectionScore,
    Document,
    MatchCounts,
    PiiType,
    Span,
    SpanMatchRule,
)
from pii_toolbox.evaluation.matcher import GreedySpanMatcher
from pii_toolbox.evaluation.scorer import (
    MatchingDetectionScorer,
    count_matches,
    overall_scores,
    score_from_counts,
)

SHARED_TYPES = frozenset(PiiType) - {PiiType.IDENTIFIER}
_BLIND_TYPE = PiiType.PERSON


@dataclass(frozen=True, slots=True)
class ExtendedDetectionReport:
    """A detection report plus the rows and overalls that depend on the dataset.

    `base` is the shared report. `shared_overall_micro` and `shared_macro` cover the five
    types every dataset has, so IDENTIFIER and extra labels never move them.
    `overall_with_identifier_micro` and `overall_with_identifier_macro` are present only for
    TAB. `extra_rows` holds one score per dataset label with no common type (TAB's DEM and
    MISC). `by_type` omits IDENTIFIER outside TAB.
    """

    base: DetectionReport
    by_type: Mapping[PiiType, DetectionScore | None]
    shared_overall_micro: DetectionScore
    shared_macro: DetectionScore
    overall_with_identifier_micro: DetectionScore | None
    overall_with_identifier_macro: DetectionScore | None
    extra_rows: Mapping[str, DetectionScore]


def _overlaps(span: Span, answer: AnswerSpan) -> bool:
    return min(span.end, answer.end) > max(span.start, answer.start)


def _extra_row_counts(
    results: Sequence[tuple[Document, Sequence[Span], Sequence[AnswerSpan]]], label: str, rule: SpanMatchRule
) -> MatchCounts:
    """Matches predictions of any type to the answers carrying a label with no common type.

    A prediction that overlaps no answer at all is a false positive; one that overlaps only
    answers of other labels says nothing about this label and is left out.
    """
    matcher = GreedySpanMatcher()
    true_positives = false_positives = false_negatives = 0
    for _, predicted, answers in results:
        label_answers = [replace(a, pii_type=_BLIND_TYPE) for a in answers if a.pii_type is None and a.label == label]
        blind_predictions = [replace(span, pii_type=_BLIND_TYPE) for span in predicted]
        counts = matcher.match(blind_predictions, label_answers, rule)[_BLIND_TYPE]
        true_positives += counts.true_positives
        false_negatives += counts.false_negatives
        false_positives += sum(1 for span in predicted if not any(_overlaps(span, a) for a in answers))
    return MatchCounts(true_positives, false_positives, false_negatives)


def score_with_extras(
    results: Iterable[tuple[Document, Sequence[Span], Sequence[AnswerSpan]]],
    rule: SpanMatchRule,
    *,
    dataset: str,
    reported_types: frozenset[PiiType] = frozenset(PiiType),
) -> ExtendedDetectionReport:
    """Scores one dataset; `dataset` is "tab" or "mapa" and decides which extra scores exist."""
    materialised = list(results)
    is_tab = dataset.lower() == "tab"
    base = MatchingDetectionScorer().score(materialised, rule, reported_types=reported_types)
    per_type, _ = count_matches(materialised, rule)
    _, shared_micro, shared_macro = overall_scores(per_type, SHARED_TYPES, reported_types)

    with_identifier_micro = with_identifier_macro = None
    by_type = dict(base.by_type)
    if is_tab:
        with_identifier_micro, with_identifier_macro = base.overall_micro, base.macro
    else:
        del by_type[PiiType.IDENTIFIER]

    extra_rows: dict[str, DetectionScore] = {}
    if is_tab:
        labels = sorted({a.label for _, _, answers in materialised for a in answers if a.pii_type is None})
        for label in labels:
            extra_rows[label] = score_from_counts(_extra_row_counts(materialised, label, rule))

    return ExtendedDetectionReport(
        base=base,
        by_type=by_type,
        shared_overall_micro=shared_micro,
        shared_macro=shared_macro,
        overall_with_identifier_micro=with_identifier_micro,
        overall_with_identifier_macro=with_identifier_macro,
        extra_rows=extra_rows,
    )
