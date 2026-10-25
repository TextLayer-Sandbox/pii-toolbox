"""Runnable round trip: mask a text, let a stand-in model edit it, restore the originals.

Run from the repository root with `uv run python examples/round_trip.py`.
"""

import re

from pii_toolbox.core import Detector, Document, EntityGroup, IdentityResolver, PiiType, Span
from pii_toolbox.session import AngleBracketFormat, InMemoryPlaceholderStore, Session

KNOWN_NAMES = ("David Parker", "Dave")
EMAIL_PATTERN = re.compile(r"[\w.]+@[\w.]+\.\w+")


class NameAndEmailDetector(Detector):
    """Finds a fixed list of names and anything shaped like an email address."""

    @property
    def name(self) -> str:
        return "name-and-email"

    @property
    def languages(self) -> frozenset[str] | None:
        return None

    @property
    def detects(self) -> frozenset[PiiType]:
        return frozenset({PiiType.PERSON, PiiType.IDENTIFIER})

    def load_models(self) -> None:
        pass

    def detect(self, document: Document) -> list[Span]:
        spans = [
            Span(match.start(), match.end(), PiiType.IDENTIFIER, "EMAIL_ADDRESS", (self.name,), None)
            for match in EMAIL_PATTERN.finditer(document.text)
        ]
        for name in KNOWN_NAMES:
            for match in re.finditer(rf"\b{re.escape(name)}\b", document.text):
                spans.append(Span(match.start(), match.end(), PiiType.PERSON, "PERSON", (self.name,), None))
        return sorted(spans, key=lambda span: span.start)


class SinglePersonResolver(IdentityResolver):
    """Treats every detected person in a text as one individual, named by the longest form."""

    @property
    def name(self) -> str:
        return "single-person"

    def resolve(self, document: Document, spans: list[Span]) -> list[EntityGroup]:
        people = tuple(span for span in spans if span.pii_type is PiiType.PERSON)
        if not people:
            return []
        return [EntityGroup(people, max(people, key=lambda span: span.end - span.start))]


def stand_in_model(text: str) -> str:
    """Echoes the text with a prefix and one phrase changed, as a language model might."""
    return "Summary: " + text.replace("wrote to", "emailed")


def main() -> None:
    session = Session(
        NameAndEmailDetector(),
        InMemoryPlaceholderStore(),
        AngleBracketFormat(),
        "en",
        resolver=SinglePersonResolver(),
    )
    with session:
        masked = session.mask("David Parker wrote to dave@acme.example. Dave replied.")
        reply = stand_in_model(masked.masked_text)
        restored = session.restore(reply)

    print(f"masked: {masked.masked_text}")
    print(f"reply: {reply}")
    print(f"restored: {restored.restored_text}")


if __name__ == "__main__":
    main()
