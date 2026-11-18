"""Joins detected names that no group holds to the one group holding a name with the same standard given name, using a nickname list."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pii_toolbox.coref.groups import TITLES
from pii_toolbox.coref.nicknames import DEFAULT_NICKNAME_LIST_PATH, load_nickname_list
from pii_toolbox.core import Document, PersonMentions, PiiType, Span


@dataclass(frozen=True)
class NicknameConfig:
    """Chooses the nickname list file, where a `list_path` of None turns the step off."""

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
    """Returns the groups with each unlinked detected name added to the one group it matches."""
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
            if any(other is not None for other in forms)
            and all(other is None or _forms_match(form, other) for other in forms)
        ]
        if len(matching) == 1:
            joined[matching[0]].append(span)
            group_forms[matching[0]].append(form)
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
