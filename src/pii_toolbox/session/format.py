"""Angle-bracket placeholder format used when masking text and when restoring model output."""

import re

from pii_toolbox.core import PiiType, Placeholder, PlaceholderFormat, PlaceholderToken

_LABEL_ALTERNATIVES = "|".join(pii_type.value for pii_type in PiiType)
# A type name, underscore and number, in any case, with optional angle brackets.
# A bracketed token matches anywhere; a bare one must not follow a letter, digit or underscore.
_TOKEN_PATTERN = re.compile(
    rf"(?:<|(?<![A-Za-z0-9_]))(?P<label>{_LABEL_ALTERNATIVES})_(?P<number>\d+)>?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


class AngleBracketFormat(PlaceholderFormat):
    """Writes placeholders as <LABEL_N> and reads them back case-insensitively, with or without brackets."""

    def render(self, placeholder: Placeholder) -> str:
        return f"<{placeholder.label}_{placeholder.number}>"

    def parse(self, text: str) -> list[PlaceholderToken]:
        return [
            PlaceholderToken(
                start=match.start(),
                end=match.end(),
                placeholder=Placeholder(match["label"].upper(), int(match["number"])),
            )
            for match in _TOKEN_PATTERN.finditer(text)
        ]
