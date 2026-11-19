"""One case's masking context: detects PII, swaps it for placeholders and discards the mapping on close."""

import logging
import uuid
from bisect import bisect_right
from collections.abc import Sequence
from types import TracebackType
from typing import Self

from pii_toolbox.core import (
    Detector,
    Document,
    PersonMentions,
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

_PLACEHOLDER_TEXT_SOURCE = "placeholder-look-alike"


def _placeholder_look_alike_spans(text: str, placeholder_format: PlaceholderFormat) -> list[Span]:
    """Spans over every stretch of `text` that the placeholder format reads as a placeholder."""
    return [
        Span(token.start, token.end, PiiType.IDENTIFIER, PiiType.IDENTIFIER.value, (_PLACEHOLDER_TEXT_SOURCE,), None)
        for token in placeholder_format.parse(text)
    ]


def _remap_person_mentions(people: list[PersonMentions], merged_spans: list[Span]) -> list[PersonMentions]:
    """Updates each person's group after merging: a span that was swallowed is swapped for the merged span containing it, then non-person spans and emptied groups are dropped."""
    merged_starts = [span.start for span in merged_spans]

    def containing(span: Span) -> Span:
        return merged_spans[bisect_right(merged_starts, span.start) - 1]

    remapped: list[PersonMentions] = []
    for person in people:
        members = list(dict.fromkeys(containing(span) for span in person.spans))
        members = [span for span in members if span.pii_type is PiiType.PERSON]
        if not members:
            continue
        canonical = containing(person.canonical)
        if canonical not in members:
            canonical = max(members, key=lambda span: span.end - span.start)
        remapped.append(PersonMentions(tuple(members), canonical))
    return remapped


def _rewrite_text(text: str, placeholder_for: dict[Span, Placeholder], placeholder_format: PlaceholderFormat) -> str:
    """Replaces every span of `text` with its rendered placeholder."""
    masked = text
    # Replacing from the end keeps the earlier offsets valid.
    for span in sorted(placeholder_for, key=lambda span: span.start, reverse=True):
        masked = masked[: span.start] + placeholder_format.render(placeholder_for[span]) + masked[span.end :]
    return masked


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
        if detector.languages is not None and language not in detector.languages:
            raise ValueError(f"detector {detector.name} does not support language {language}")
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
        spans, people = self._find_spans(document)
        placeholder_for, minted = self._assign_placeholders(text, spans, people)
        masked = _rewrite_text(text, placeholder_for, self._format)

        logger.info(
            "masked text %s: %d spans replaced, minted %s",
            document.id,
            len(placeholder_for),
            [self._format.render(placeholder) for placeholder in minted],
        )
        return MaskResult(masked, tuple(minted), placeholder_for)

    def _find_spans(self, document: Document) -> tuple[list[Span], list[PersonMentions]]:
        """Detects, resolves and merges the spans of a document and returns them with the remapped person mentions."""
        detected = self._detector.detect(document)
        people = self._resolver.resolve(document, detected) if self._resolver is not None else []
        person_spans = [span for person in people for span in person.spans]
        merged_spans = merge_overlapping(
            [*detected, *person_spans, *_placeholder_look_alike_spans(document.text, self._format)]
        )
        return merged_spans, _remap_person_mentions(people, merged_spans)

    def _assign_placeholders(
        self, text: str, spans: list[Span], people: list[PersonMentions]
    ) -> tuple[dict[Span, Placeholder], list[Placeholder]]:
        """Looks up or mints a placeholder for every span, registers person aliases and returns the span mapping with the newly minted placeholders."""
        placeholder_for: dict[Span, Placeholder] = {}
        minted: list[Placeholder] = []

        linked_spans = {span for person in people for span in person.spans}
        groups = [(person.canonical, person.spans) for person in people]
        groups.extend((span, (span,)) for span in spans if span not in linked_spans)

        for fullest_span, group_spans in groups:
            label = fullest_span.pii_type.value
            fullest_name = text[fullest_span.start : fullest_span.end]
            names = [text[span.start : span.end] for span in group_spans]
            placeholder = self._known_placeholder(label, fullest_name, names)
            if placeholder is None:
                placeholder, was_minted = self._store.get_or_mint(self._id, label, fullest_name)
                if was_minted:
                    minted.append(placeholder)
            for name in names:
                if self._store.placeholder_for(self._id, label, name) is None:
                    self._store.alias(self._id, name, placeholder)
            for span in group_spans:
                placeholder_for[span] = placeholder

        return placeholder_for, minted

    def _known_placeholder(self, label: str, fullest_name: str, names: Sequence[str]) -> Placeholder | None:
        """Returns the stored placeholder a group of names already belongs to, or None when the group needs a new one."""
        for_fullest = self._store.placeholder_for(self._id, label, fullest_name)
        if for_fullest is not None:
            return for_fullest
        fullest_in_group = [
            candidate
            for name in names
            if (candidate := self._store.placeholder_for(self._id, label, name)) is not None
            and self._store.original(self._id, candidate) == name
        ]
        if fullest_in_group:
            return min(fullest_in_group, key=lambda candidate: candidate.number)
        return None

    def restore(self, text: str) -> RestoreResult:
        minted_labels = self._store.labels(self._id)
        leftovers: list[str] = []
        restored = text

        # Substituting from the last token to the first keeps the earlier offsets valid.
        for token in sorted(self._format.parse(text), key=lambda token: token.start, reverse=True):
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
