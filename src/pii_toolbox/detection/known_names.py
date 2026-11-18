"""Detector for a configured list of names that are always recognised as people."""

import re
from collections.abc import Iterable

from pii_toolbox.core import Detector, Document, PiiType, Span


class KnownNamesDetector(Detector):
    """Reports each whole-word occurrence of a listed name as a person, requiring a leading capital unless `case_insensitive` is set."""

    def __init__(self, names: Iterable[str], case_insensitive: bool = False) -> None:
        unique_names = sorted({name.strip() for name in names if name.strip()}, key=len, reverse=True)
        self._case_insensitive = case_insensitive
        self._pattern = (
            re.compile(
                r"(?<!\w)(?:" + "|".join(re.escape(name) for name in unique_names) + r")(?!\w)",
                re.IGNORECASE,
            )
            if unique_names
            else None
        )

    @property
    def name(self) -> str:
        return "known_names"

    @property
    def languages(self) -> frozenset[str] | None:
        return None

    @property
    def detects(self) -> frozenset[PiiType]:
        return frozenset({PiiType.PERSON})

    def load_models(self) -> None:
        """Does nothing: the detector has no models."""

    def detect(self, document: Document) -> list[Span]:
        if self._pattern is None:
            return []
        return [
            Span(
                start=match.start(),
                end=match.end(),
                pii_type=PiiType.PERSON,
                detector_label="KNOWN_NAME",
                source=(self.name,),
                score=1.0,
            )
            for match in self._pattern.finditer(document.text)
            if self._case_insensitive or match.group()[0].isupper()
        ]
