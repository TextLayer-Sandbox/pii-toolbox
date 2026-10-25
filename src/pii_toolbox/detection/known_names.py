"""Detector for a configured list of names that are always recognised as people.

Matching is exact and whole-word: no fuzzy or typo matching, and a name inside a longer
word is ignored. No model is involved, so nothing is loaded and nothing touches the network.
"""

import re
from collections.abc import Iterable
from pathlib import Path

from pii_toolbox.core import Detector, Document, PiiType, Span


def read_names_file(path: Path) -> list[str]:
    """Reads one name per line, skipping blank lines."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class KnownNamesDetector(Detector):
    """Reports every whole-word occurrence of a listed name as a person.

    By default a name matches only with the letter case it is listed in, so listed 'Rose'
    matches 'Rose' and not 'rose'. With `case_insensitive` it matches in any letter case.
    Multi-word names match as written, with the listed spacing.
    """

    def __init__(self, names: Iterable[str], case_insensitive: bool = False) -> None:
        unique_names = sorted({name.strip() for name in names if name.strip()}, key=len, reverse=True)
        self._pattern = (
            re.compile(
                r"(?<!\w)(?:" + "|".join(re.escape(name) for name in unique_names) + r")(?!\w)",
                re.IGNORECASE if case_insensitive else 0,
            )
            if unique_names
            else None
        )

    @classmethod
    def from_file(cls, path: Path, case_insensitive: bool = False) -> "KnownNamesDetector":
        return cls(read_names_file(path), case_insensitive)

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
        ]
