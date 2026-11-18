"""Turns mention clusters from a co-reference model into groups of spans that refer to one person each."""

import re
from collections.abc import Sequence

from pii_toolbox.core import Document, PersonMentions, PiiType, Span

RESOLVER_NAME = "maverick"

TITLES = frozenset({"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "sir", "madam", "lord", "lady"})
_CAPITALISED_WORD = re.compile(r"[A-Z][\w'’-]*\.?")
# Capitalised only because they open a sentence.
_CAPITALISED_NON_NAMES = frozenset(
    {"i", "he", "she", "it", "we", "they", "you", "him", "her", "his", "hers", "them", "their", "the", "a", "an", "this", "that"}
)


def build_groups(
    document: Document,
    detected_spans: Sequence[Span],
    mention_clusters: Sequence[Sequence[tuple[int, int]]],
) -> list[PersonMentions]:
    """Returns one group per cluster that holds a detected PERSON span, ordered by first position; raises ValueError for a mention outside the document text."""
    person_spans = [span for span in detected_spans if span.pii_type is PiiType.PERSON]
    groups = []
    for cluster in mention_clusters:
        group = _group_from_cluster(document, person_spans, cluster)
        if group is not None:
            groups.append(group)
    groups.sort(key=lambda group: group.spans[0].start)
    return groups


def _group_from_cluster(
    document: Document,
    person_spans: Sequence[Span],
    cluster: Sequence[tuple[int, int]],
) -> PersonMentions | None:
    detected_in_cluster: dict[tuple[int, int], Span] = {}
    added: dict[tuple[int, int], Span] = {}
    for start, end in cluster:
        if not 0 <= start < end <= len(document.text):
            raise ValueError(f"mention ({start}, {end}) is outside document {document.id!r}")
        overlapping = {(span.start, span.end): span for span in person_spans if span.start < end and start < span.end}
        if len(overlapping) > 1:
            continue
        detected_in_cluster.update(overlapping)
        if not overlapping and _is_name_form(document.text[start:end]):
            added[(start, end)] = Span(
                start=start,
                end=end,
                pii_type=PiiType.PERSON,
                detector_label=PiiType.PERSON.value,
                source=(RESOLVER_NAME,),
                score=None,
            )
    if not detected_in_cluster:
        return None
    spans = sorted([*detected_in_cluster.values(), *added.values()], key=lambda span: (span.start, span.end))
    canonical = max(spans, key=lambda span: span.end - span.start)
    return PersonMentions(spans=tuple(spans), canonical=canonical)


def _is_name_form(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    if words[0].rstrip(".").lower() in TITLES:
        return len(words) > 1 and all(_CAPITALISED_WORD.fullmatch(word) for word in words[1:])
    return all(
        _CAPITALISED_WORD.fullmatch(word) and word.rstrip(".").lower() not in _CAPITALISED_NON_NAMES
        for word in words
    )
