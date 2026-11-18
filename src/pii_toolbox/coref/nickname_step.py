"""Joins detected names that the co-reference model left unlinked to a group through a nickname list.

A name's standard form is its given name looked up in the nickname list (dave -> david). An
unlinked detected PERSON name joins the one group holding a name with the same standard form
and a compatible surname. Nicknames never link to each other directly: two names that share a
standard form are linked only when a group already holds one of them. The step reads the
document text to find the names and returns groups of positions only.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pii_toolbox.coref.groups import TITLES
from pii_toolbox.coref.nicknames import DEFAULT_NICKNAME_LIST_PATH, load_nickname_list
from pii_toolbox.core import Document, PersonMentions, PiiType, Span


@dataclass(frozen=True)
class NicknameConfig:
    """Chooses the nickname list: the committed default, another file of the same shape, or none.

    A `list_path` of None turns the step off.
    """

    list_path: Path | None = DEFAULT_NICKNAME_LIST_PATH

    def load(self) -> dict[str, str] | None:
        """Returns the validated nickname-to-standard-name mapping, or None when the step is off."""
        if self.list_path is None:
            return None
        return load_nickname_list(self.list_path)


def apply_nickname_step(
    document: Document,
    detected_spans: Sequence[Span],
    groups: Sequence[PersonMentions],
    nicknames: Mapping[str, str] | None,
) -> list[PersonMentions]:
    """Returns the groups with each unlinked detected name added to the group it matches.

    A name already in a group stays there. A name that matches more than one group, or none,
    stays unlinked. With `nicknames` None (the list turned off) the groups are returned unchanged.
    """
    if nicknames is None:
        return list(groups)
    grouped_positions = {(span.start, span.end) for group in groups for span in group.spans}
    unlinked = [
        span
        for span in detected_spans
        if span.pii_type is PiiType.PERSON and (span.start, span.end) not in grouped_positions
    ]
    group_forms = [
        [_name_form(document.text[span.start : span.end], nicknames) for span in group.spans] for group in groups
    ]
    joined: list[list[Span]] = [[] for _ in groups]
    for span in unlinked:
        form = _name_form(document.text[span.start : span.end], nicknames)
        if form is None:
            continue
        matching = [
            index
            for index, forms in enumerate(group_forms)
            if any(other is not None and _forms_match(form, other) for other in forms)
        ]
        if len(matching) == 1:
            joined[matching[0]].append(span)
    return [
        PersonMentions(spans=tuple(sorted([*group.spans, *extra], key=lambda s: (s.start, s.end))), canonical=group.canonical)
        if extra
        else group
        for group, extra in zip(groups, joined)
    ]


def _name_form(text: str, nicknames: Mapping[str, str]) -> tuple[str, str | None] | None:
    """Returns (standard given name, lower-case surname or None); None when no given name is present."""
    words = [word.strip(".,;:").lower() for word in text.split()]
    words = [word for word in words if word]
    if words and words[0] in TITLES:
        if len(words) < 3:
            return None
        words = words[1:]
    if not words:
        return None
    given = words[0]
    return nicknames.get(given, given), (words[-1] if len(words) > 1 else None)


def _forms_match(first: tuple[str, str | None], second: tuple[str, str | None]) -> bool:
    if first[0] != second[0]:
        return False
    return first[1] is None or second[1] is None or first[1] == second[1]
