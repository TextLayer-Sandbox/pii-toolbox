"""Measures whether identity linking gives each gold PERSON entity one placeholder, scored per annotator."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields

from pii_toolbox.core import (
    AnnotatedDocument,
    AnswerSpan,
    Detector,
    Document,
    IdentityResolver,
    PiiType,
    Placeholder,
    ReidentificationRisk,
    Span,
)
from pii_toolbox.session import AngleBracketFormat, InMemoryPlaceholderStore, Session

_MASKED_RISKS = frozenset({ReidentificationRisk.DIRECT, ReidentificationRisk.QUASI})
_RATES = {
    "linked_rate_multi_form_entities": ("multi_form_entities_linked", "multi_form_entities"),
    "linked_rate_multi_mention_entities": ("multi_mention_entities_linked", "multi_mention_entities"),
    "wrongly_merged_rate_of_placeholders": ("wrongly_merged_placeholders", "placeholders"),
}


@dataclass(frozen=True, slots=True)
class LinkingCounts:
    """Linking counts for one annotator. An entity is linked when all its detected mentions share one placeholder."""

    detected_person_spans: int = 0
    matched_spans: int = 0
    unmatched_spans: int = 0
    multi_mention_entities: int = 0
    multi_mention_entities_linked: int = 0
    multi_form_entities: int = 0
    multi_form_entities_linked: int = 0
    placeholders: int = 0
    wrongly_merged_placeholders: int = 0

    def __add__(self, other: "LinkingCounts") -> "LinkingCounts":
        return LinkingCounts(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))

    def rate(self, name: str) -> float | None:
        """None when the denominator is zero."""
        numerator, denominator = _RATES[name]
        total = getattr(self, denominator)
        return getattr(self, numerator) / total if total else None


@dataclass(frozen=True, slots=True)
class LinkingMeasurement:
    """Counts per annotator with the resolver off and on."""

    off: Mapping[str, LinkingCounts]
    on: Mapping[str, LinkingCounts]


@dataclass(frozen=True, slots=True)
class _GoldMention:
    start: int
    end: int
    entity_id: str
    form: str


def _overlap(span: Span, mention: _GoldMention) -> int:
    return min(span.end, mention.end) - max(span.start, mention.start)


def _gold_by_annotator(document: Document, answers: Sequence[AnswerSpan]) -> dict[str, list[_GoldMention]]:
    gold: dict[str, list[_GoldMention]] = defaultdict(list)
    for answer in answers:
        if answer.pii_type is not PiiType.PERSON or answer.risk not in _MASKED_RISKS:
            continue
        if answer.annotator is None or answer.entity_id is None:
            continue
        form = document.text[answer.start : answer.end].casefold()
        gold[answer.annotator].append(_GoldMention(answer.start, answer.end, answer.entity_id, form))
    return gold


def _best_mention(span: Span, mentions: Sequence[_GoldMention]) -> _GoldMention | None:
    """Largest overlap wins; earliest on a tie."""
    overlapping = [mention for mention in mentions if _overlap(span, mention) > 0]
    if not overlapping:
        return None
    return max(overlapping, key=lambda mention: (_overlap(span, mention), -mention.start))


def count_linking(
    document: Document,
    answers: Sequence[AnswerSpan],
    detected_spans: Sequence[Span],
    placeholder_by_span: Mapping[Span, Placeholder],
) -> dict[str, LinkingCounts]:
    """Linking counts per annotator for one document."""
    person_spans = [span for span in detected_spans if span.pii_type is PiiType.PERSON]
    counts = {}
    for annotator, mentions in _gold_by_annotator(document, answers).items():
        forms_by_entity: dict[str, set[str]] = defaultdict(set)
        for mention in mentions:
            forms_by_entity[mention.entity_id].add(mention.form)
        placeholders_by_entity: dict[str, set[Placeholder]] = defaultdict(set)
        entities_by_placeholder: dict[Placeholder, set[str]] = defaultdict(set)
        matched = 0
        mention_counts: dict[str, int] = defaultdict(int)
        for span in person_spans:
            best = _best_mention(span, mentions)
            if best is None:
                continue
            matched += 1
            mention_counts[best.entity_id] += 1
            placeholder = placeholder_by_span[span]
            placeholders_by_entity[best.entity_id].add(placeholder)
            entities_by_placeholder[placeholder].add(best.entity_id)
        multi_mention = [entity for entity, count in mention_counts.items() if count >= 2]
        multi_form = [entity for entity in multi_mention if len(forms_by_entity[entity]) >= 2]
        counts[annotator] = LinkingCounts(
            detected_person_spans=len(person_spans),
            matched_spans=matched,
            unmatched_spans=len(person_spans) - matched,
            multi_mention_entities=len(multi_mention),
            multi_mention_entities_linked=sum(len(placeholders_by_entity[entity]) == 1 for entity in multi_mention),
            multi_form_entities=len(multi_form),
            multi_form_entities_linked=sum(len(placeholders_by_entity[entity]) == 1 for entity in multi_form),
            placeholders=len(entities_by_placeholder),
            wrongly_merged_placeholders=sum(len(entities) >= 2 for entities in entities_by_placeholder.values()),
        )
    return counts


class _FixedSpansDetector(Detector):
    name = "fixed_spans"
    languages = None
    detects = frozenset(PiiType)

    def __init__(self, spans: Sequence[Span]) -> None:
        self._spans = list(spans)

    def load_models(self) -> None:
        pass

    def detect(self, document: Document) -> list[Span]:
        return list(self._spans)


def _mask_placeholders(
    document: Document, spans: Sequence[Span], resolver: IdentityResolver | None
) -> Mapping[Span, Placeholder]:
    session = Session(_FixedSpansDetector(spans), InMemoryPlaceholderStore(), AngleBracketFormat(), document.language, resolver)
    with session:
        return session.mask(document.text).placeholder_by_span


def _add_counts(total: dict[str, LinkingCounts], new: Mapping[str, LinkingCounts]) -> None:
    for annotator, counts in new.items():
        total[annotator] = total.get(annotator, LinkingCounts()) + counts


def measure_linking(
    documents: Iterable[AnnotatedDocument], detector: Detector, resolver: IdentityResolver
) -> LinkingMeasurement:
    """The detector must have its models loaded."""
    off: dict[str, LinkingCounts] = {}
    on: dict[str, LinkingCounts] = {}
    for annotated in documents:
        document = annotated.document
        spans = detector.detect(document)
        for total, used_resolver in ((off, None), (on, resolver)):
            placeholders = _mask_placeholders(document, spans, used_resolver)
            _add_counts(total, count_linking(document, annotated.answers, spans, placeholders))
    return LinkingMeasurement(off, on)


def _counts_to_dict(counts: LinkingCounts) -> dict[str, int | float | None]:
    content: dict[str, int | float | None] = {f.name: getattr(counts, f.name) for f in fields(counts)}
    for name in _RATES:
        content[name] = counts.rate(name)
    return content


def linking_report(by_annotator: Mapping[str, LinkingCounts]) -> dict:
    """Per-annotator, pooled and macro-averaged counts and rates."""
    pooled = sum(by_annotator.values(), LinkingCounts())
    macro: dict[str, float | None] = {}
    for name in _RATES:
        rates = [rate for rate in (counts.rate(name) for counts in by_annotator.values()) if rate is not None]
        macro[name] = sum(rates) / len(rates) if rates else None
    return {
        "per_annotator": {annotator: _counts_to_dict(counts) for annotator, counts in sorted(by_annotator.items())},
        "pooled": _counts_to_dict(pooled),
        "macro_average_rates": macro,
    }
