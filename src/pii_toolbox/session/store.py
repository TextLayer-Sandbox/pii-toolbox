"""Placeholder store that keeps per-session mappings in process memory and nowhere else."""

from dataclasses import dataclass, field

from pii_toolbox.core import Placeholder, PlaceholderStore


@dataclass(slots=True, repr=False)
class _SessionMappings:
    """Everything one session knows; dropped as a whole when the session is discarded."""

    last_number_by_label: dict[str, int] = field(default_factory=dict)
    placeholder_by_original: dict[tuple[str, str], Placeholder] = field(default_factory=dict)
    forms_by_placeholder: dict[Placeholder, list[str]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"_SessionMappings(placeholders={len(self.forms_by_placeholder)})"


class InMemoryPlaceholderStore(PlaceholderStore):
    """Placeholder store keyed by session id; originals are never persisted or logged."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionMappings] = {}

    def __repr__(self) -> str:
        return f"InMemoryPlaceholderStore(sessions={len(self._sessions)})"

    def get_or_mint(self, session_id: str, label: str, original: str) -> tuple[Placeholder, bool]:
        mappings = self._sessions.setdefault(session_id, _SessionMappings())
        existing = mappings.placeholder_by_original.get((label, original))
        if existing is not None:
            return existing, False
        number = mappings.last_number_by_label.get(label, 0) + 1
        mappings.last_number_by_label[label] = number
        placeholder = Placeholder(label, number)
        mappings.placeholder_by_original[(label, original)] = placeholder
        mappings.forms_by_placeholder[placeholder] = [original]
        return placeholder, True

    def placeholder_for(self, session_id: str, label: str, original: str) -> Placeholder | None:
        mappings = self._sessions.get(session_id)
        if mappings is None:
            return None
        return mappings.placeholder_by_original.get((label, original))

    def alias(self, session_id: str, original: str, placeholder: Placeholder) -> None:
        mappings = self._sessions.setdefault(session_id, _SessionMappings())
        forms = mappings.forms_by_placeholder.get(placeholder)
        if forms is None:
            return
        mappings.placeholder_by_original[(placeholder.label, original)] = placeholder
        if original not in forms:
            forms.append(original)

    def original(self, session_id: str, placeholder: Placeholder) -> str | None:
        mappings = self._sessions.get(session_id)
        if mappings is None:
            return None
        forms = mappings.forms_by_placeholder.get(placeholder)
        if not forms:
            return None
        return max(forms, key=len)

    def labels(self, session_id: str) -> frozenset[str]:
        mappings = self._sessions.get(session_id)
        if mappings is None:
            return frozenset()
        return frozenset(mappings.last_number_by_label)

    def discard(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
