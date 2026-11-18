"""One case's masking context: detects PII, swaps it for placeholders and discards the mapping on close."""

import logging
import uuid
from bisect import bisect_right
from types import TracebackType
from typing import Self

from pii_toolbox.core import (
    Detector,
    Document,
    EntityGroup,
    IdentityResolver,
    MaskResult,
    PiiType,
    Placeholder,
    PlaceholderFormat,
    PlaceholderStore,
    RedactionSession,
    RestoreResult,
    Span,
)
from pii_toolbox.detection.union import merge_overlapping

logger = logging.getLogger(__name__)

PLACEHOLDER_TEXT_LABEL = "PLACEHOLDER_TEXT"
_PLACEHOLDER_TEXT_SOURCE = "placeholder-look-alike"


def _placeholder_look_alike_spans(text: str, placeholder_format: PlaceholderFormat) -> list[Span]:
    """Spans over every stretch of `text` written exactly as the placeholder format renders a placeholder."""
    return [
        Span(token.start, token.end, PiiType.IDENTIFIER, PLACEHOLDER_TEXT_LABEL, (_PLACEHOLDER_TEXT_SOURCE,), None)
        for token in placeholder_format.parse(text)
        if text[token.start : token.end] == placeholder_format.render(token.placeholder)
    ]


def _merge_overlaps(
    spans: list[Span], groups: list[EntityGroup], extra_spans: list[Span]
) -> tuple[list[Span], list[EntityGroup]]:
    """Returns the spans with overlaps merged and the groups rewritten onto the merged spans."""
    group_spans = [span for group in groups for span in group.spans]
    merged = merge_overlapping([*spans, *group_spans, *extra_spans])
    starts = [span.start for span in merged]

    def containing(span: Span) -> Span:
        return merged[bisect_right(starts, span.start) - 1]

    remapped: list[EntityGroup] = []
    for group in groups:
        members = list(dict.fromkeys(containing(span) for span in group.spans))
        members = [span for span in members if span.pii_type is PiiType.PERSON]
        if not members:
            continue
        canonical = containing(group.canonical)
        if canonical not in members:
            canonical = max(members, key=lambda span: span.end - span.start)
        remapped.append(EntityGroup(tuple(members), canonical))
    return merged, remapped


class Session(RedactionSession):
    """Masks every text of one case against a shared store, so equal originals share a placeholder."""

    def __init__(
        self,
        detector: Detector,
        store: PlaceholderStore,
        placeholder_format: PlaceholderFormat,
        language: str,
        resolver: IdentityResolver | None = None,
        session_id: str | None = None,
    ) -> None:
        self._id = session_id if session_id is not None else uuid.uuid4().hex
        self._language = language
        self._store = store
        self._format = placeholder_format
        self._detector = detector
        self._resolver = resolver
        self._text_counter = 0

    @property
    def id(self) -> str:
        return self._id

    @property
    def language(self) -> str:
        return self._language

    @property
    def store(self) -> PlaceholderStore:
        return self._store

    @property
    def format(self) -> PlaceholderFormat:
        return self._format

    @property
    def detector(self) -> Detector:
        return self._detector

    @property
    def resolver(self) -> IdentityResolver | None:
        return self._resolver

    def mask(self, text: str) -> MaskResult:
        self._text_counter += 1
        document = Document(f"{self._id}-{self._text_counter}", text, self._language)
        spans = self._detector.detect(document)

        placeholder_for: dict[Span, Placeholder] = {}
        minted: list[Placeholder] = []

        def get_or_mint(label: str, original: str) -> Placeholder:
            placeholder, was_minted = self._store.get_or_mint(self._id, label, original)
            if was_minted:
                minted.append(placeholder)
            return placeholder

        groups = self._resolver.resolve(document, spans) if self._resolver is not None else []
        spans, groups = _merge_overlaps(spans, groups, _placeholder_look_alike_spans(text, self._format))

        for group in groups:
            fullest = text[group.canonical.start : group.canonical.end]
            placeholder = get_or_mint(group.canonical.detector_label, fullest)
            for span in group.spans:
                if span != group.canonical:
                    self._store.alias(self._id, text[span.start : span.end], placeholder)
                placeholder_for[span] = placeholder

        for span in spans:
            if span not in placeholder_for:
                placeholder_for[span] = get_or_mint(span.detector_label, text[span.start : span.end])

        masked = text
        # Replacing from the end keeps the earlier offsets valid.
        for span in sorted(placeholder_for, key=lambda s: s.start, reverse=True):
            masked = masked[: span.start] + self._format.render(placeholder_for[span]) + masked[span.end :]

        logger.info(
            "masked text %s: %d spans replaced, minted %s",
            document.id,
            len(placeholder_for),
            [self._format.render(placeholder) for placeholder in minted],
        )
        return MaskResult(masked, tuple(minted), placeholder_for)

    def restore(self, text: str) -> RestoreResult:
        minted_labels = self._store.labels(self._id)
        leftovers: list[str] = []
        restored = text

        # Substituting from the last token to the first keeps the earlier offsets valid.
        for token in sorted(self._format.parse(text), key=lambda t: t.start, reverse=True):
            original = self._store.original(self._id, token.placeholder)
            if original is not None:
                restored = restored[: token.start] + original + restored[token.end :]
            elif token.placeholder.label in minted_labels:
                leftovers.append(text[token.start : token.end])

        return RestoreResult(restored, tuple(reversed(leftovers)))

    def close(self) -> None:
        self._store.discard(self._id)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
