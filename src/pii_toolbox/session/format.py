"""Angle-bracket placeholder format used when masking text and when restoring model output."""

import re

from pii_toolbox.core import PiiType, Placeholder, PlaceholderFormat, PlaceholderToken

_LABEL_ALTERNATIVES = "|".join(pii_type.value for pii_type in PiiType)
# Exactly the rendered form: upper-case type name, underscore, number without leading zeros, in angle brackets.
_TOKEN_PATTERN = re.compile(rf"<(?P<label>{_LABEL_ALTERNATIVES})_(?P<number>[1-9]\d*)>")


class AngleBracketFormat(PlaceholderFormat):
    """Writes placeholders as <LABEL_N> and reads back only that exact form."""

    def render(self, placeholder: Placeholder) -> str:
        return f"<{placeholder.label}_{placeholder.number}>"

    def parse(self, text: str) -> list[PlaceholderToken]:
        return [
            PlaceholderToken(
                start=match.start(),
                end=match.end(),
                placeholder=Placeholder(match["label"], int(match["number"])),
            )
            for match in _TOKEN_PATTERN.finditer(text)
        ]
