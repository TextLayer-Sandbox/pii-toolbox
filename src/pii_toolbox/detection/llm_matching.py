"""Maps items quoted back by a language model onto character positions in the source text.

A model repeats values from the text it was given, often with straightened or curled quotes,
collapsed whitespace or changed case, so matching is done on a normalised copy of both
sides and the positions are translated back to the original text.
"""

from collections.abc import Sequence

from pii_toolbox.core import LabelMap, PiiType, Span

_QUOTE_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
    }
)

DEFAULT_LABEL_MAP: LabelMap = {pii_type.value: pii_type for pii_type in PiiType}


def _normalise(text: str) -> tuple[str, list[int], list[int]]:
    """Returns the normalised text with, per normalised character, its original start and end."""
    characters: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            run_end = index
            while run_end < len(text) and text[run_end].isspace():
                run_end += 1
            characters.append(" ")
            starts.append(index)
            ends.append(run_end)
            index = run_end
            continue
        for character in text[index].translate(_QUOTE_TRANSLATION).lower():
            characters.append(character)
            starts.append(index)
            ends.append(index + 1)
        index += 1
    return "".join(characters), starts, ends


def match_items(
    text: str,
    items: Sequence[tuple[str, str]],
    label_map: LabelMap = DEFAULT_LABEL_MAP,
    source: str = "language-model",
) -> tuple[list[Span], list[tuple[str, str]]]:
    """Finds each (value, label) item in `text`.

    Items with the same value take successive occurrences in the order given. Items whose value
    is absent, or whose label is not in `label_map`, come back in the second list and produce
    no span. Spans are sorted by start.
    """
    normalised_text, starts, ends = _normalise(text)
    next_search_from: dict[str, int] = {}
    spans: list[Span] = []
    unmatched: list[tuple[str, str]] = []
    for value, label in items:
        pii_type = label_map.get(label)
        normalised_value = _normalise(value)[0].strip()
        if pii_type is None or not normalised_value:
            unmatched.append((value, label))
            continue
        position = normalised_text.find(normalised_value, next_search_from.get(normalised_value, 0))
        if position == -1:
            unmatched.append((value, label))
            continue
        last = position + len(normalised_value) - 1
        next_search_from[normalised_value] = last + 1
        spans.append(Span(starts[position], ends[last], pii_type, label, (source,), None))
    return sorted(spans, key=lambda span: (span.start, span.end)), unmatched
