"""Angle-bracket placeholder format used when masking text and when restoring model output."""

import re

from pii_toolbox.core import Placeholder, PlaceholderFormat, PlaceholderToken

# One definition of a whole token; brackets are optional so a model that drops them is still understood.
# A bracketed token matches wherever it sits; a bare one must not follow a letter, digit or underscore.
_TOKEN_PATTERN = re.compile(
    r"(?:<|(?<![A-Za-z0-9_]))(?P<label>[A-Za-z][A-Za-z_]*)_(?P<number>\d+)>?(?![A-Za-z0-9_])"
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
