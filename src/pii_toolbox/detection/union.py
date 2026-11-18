"""Union of several detectors with overlapping spans merged into the widest."""

from collections.abc import Iterable

from pii_toolbox.core import Detector, Document, PiiType, Span
from pii_toolbox.core import DetectorUnion as DetectorUnionInterface


def _outranks(candidate: Span, current: Span) -> bool:
    """True when `candidate` is a better source of type and label than `current`."""
    candidate_length = candidate.end - candidate.start
    current_length = current.end - current.start
    if candidate_length != current_length:
        return candidate_length > current_length
    return (candidate.score if candidate.score is not None else float("-inf")) > (
        current.score if current.score is not None else float("-inf")
    )


def _merge_group(group: list[Span]) -> Span:
    best = group[0]
    for span in group[1:]:
        if _outranks(span, best):
            best = span
    sources: list[str] = []
    for span in group:
        for name in span.source:
            if name not in sources:
                sources.append(name)
    scores = [span.score for span in group if span.score is not None]
    return Span(
        start=min(span.start for span in group),
        end=max(span.end for span in group),
        pii_type=best.pii_type,
        detector_label=best.detector_label,
        source=tuple(sources),
        score=max(scores) if scores else None,
    )


def merge_overlapping(spans: Iterable[Span]) -> list[Span]:
    """Merges every cluster of overlapping spans into one span; returns the result sorted by start."""
    merged: list[Span] = []
    group: list[Span] = []
    group_end = 0
    for span in sorted(spans, key=lambda span: (span.start, span.end)):
        # Touching spans (end == next start) stay separate.
        if group and span.start < group_end:
            group.append(span)
            group_end = max(group_end, span.end)
            continue
        if group:
            merged.append(_merge_group(group))
        group = [span]
        group_end = span.end
    if group:
        merged.append(_merge_group(group))
    return merged


class DetectorUnion(DetectorUnionInterface):
    """Runs every member detector and merges overlapping spans into one."""

    def __init__(self, members: tuple[Detector, ...]) -> None:
        self._members = tuple(members)

    @property
    def members(self) -> tuple[Detector, ...]:
        return self._members

    @property
    def name(self) -> str:
        return "+".join(member.name for member in self._members)

    @property
    def languages(self) -> frozenset[str] | None:
        supported = [member.languages for member in self._members if member.languages is not None]
        if not supported:
            return None
        return frozenset.intersection(*supported)

    @property
    def detects(self) -> frozenset[PiiType]:
        return frozenset().union(*(member.detects for member in self._members))

    def load_models(self) -> None:
        for member in self._members:
            member.load_models()

    def detect(self, document: Document) -> list[Span]:
        return merge_overlapping([span for member in self._members for span in member.detect(document)])
