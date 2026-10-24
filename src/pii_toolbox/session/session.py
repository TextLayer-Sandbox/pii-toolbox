"""One case's masking context: detects PII, swaps it for placeholders and discards the mapping on close."""

import logging
import uuid
from types import TracebackType
from typing import Self

from pii_toolbox.core import (
    Detector,
    Document,
    IdentityResolver,
    MaskResult,
    Placeholder,
    PlaceholderFormat,
    PlaceholderStore,
    RedactionSession,
    RestoreResult,
    Span,
)

logger = logging.getLogger(__name__)


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

        if self._resolver is not None:
            for group in self._resolver.resolve(document, spans):
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
        return MaskResult(masked, tuple(minted))

    def restore(self, text: str) -> RestoreResult:
        raise NotImplementedError

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
