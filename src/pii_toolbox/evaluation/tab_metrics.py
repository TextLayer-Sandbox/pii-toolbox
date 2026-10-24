"""TAB's own anonymisation metrics: entity recall on direct and quasi identifiers, and weighted precision.

The definitions follow the TAB benchmark. An entity is one annotator's set of mentions that
share an entity id. It needs masking when any of its mentions is marked DIRECT or QUASI, and
it is direct when its first mention is DIRECT. An entity counts as found only when every
mention that needs masking is covered by the predicted spans, ignoring punctuation, spaces,
function words and titles. Recall pools entities over annotators. Precision is taken over the
word tokens inside predicted spans: each token earns the share of annotators whose masking
mention fully covers it, weighted by how much information the token carries.
"""

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from pii_toolbox.core import (
    AnnotatedDocument,
    AnswerSpan,
    ReidentificationRisk,
    Span,
    TabMetricsScorer,
    TabOfficialMetrics,
)

IGNORED_PARTS_OF_SPEECH = frozenset({"ADP", "PART", "CCONJ", "DET"})
IGNORED_WORDS = frozenset({"mr", "mrs", "ms", "no", "nr", "about"})
IGNORED_CHARACTERS = frozenset(" ,.-;:/&()[]–'\" ’“”")
_WORD = re.compile(r"\w+")
_MASKING_RISKS = frozenset({ReidentificationRisk.DIRECT, ReidentificationRisk.QUASI})


@dataclass(frozen=True, slots=True)
class TokenInfo:
    """One token of a document: character bounds, part-of-speech tag and lower-cased text."""

    start: int
    end: int
    part_of_speech: str
    lower: str


TokenAnnotator = Callable[[str], Sequence[TokenInfo]]
TokenWeigher = Callable[[str, Sequence[tuple[int, int]]], Sequence[float]]


def uniform_weights(text: str, spans: Sequence[tuple[int, int]]) -> list[float]:
    return [1.0] * len(spans)


@dataclass(frozen=True, slots=True)
class _Entity:
    annotator: str | None
    mentions: tuple[tuple[int, int], ...]
    masking: tuple[bool, ...]  # per mention: does it need masking
    is_direct: bool

    @property
    def needs_masking(self) -> bool:
        return any(self.masking)


def _entities(answers: Sequence[AnswerSpan]) -> list[_Entity]:
    grouped: dict[tuple[str | None, str | None], list[AnswerSpan]] = defaultdict(list)
    for index, answer in enumerate(answers):
        # A mention without an entity id is its own entity.
        key = (answer.annotator, answer.entity_id if answer.entity_id is not None else f"#{index}")
        grouped[key].append(answer)
    entities = []
    for (annotator, _), mentions in grouped.items():
        entities.append(
            _Entity(
                annotator=annotator,
                mentions=tuple((m.start, m.end) for m in mentions),
                masking=tuple(m.risk in _MASKING_RISKS for m in mentions),
                is_direct=mentions[0].risk is ReidentificationRisk.DIRECT,
            )
        )
    return entities


class OfficialTabMetricsScorer(TabMetricsScorer):
    """Computes ER_di, ER_qi and WP; token tagging and token weighting are injected."""

    def __init__(self, annotate_tokens: TokenAnnotator, weigh_tokens: TokenWeigher = uniform_weights) -> None:
        self._annotate_tokens = annotate_tokens
        self._weigh_tokens = weigh_tokens

    def score(self, results: Iterable[tuple[AnnotatedDocument, Sequence[Span]]]) -> TabOfficialMetrics:
        direct_found = direct_total = quasi_found = quasi_total = 0
        weighted_hits = weighted_masks = 0.0
        for annotated, predicted in results:
            text = annotated.document.text
            masked = {offset for span in predicted for offset in range(span.start, span.end)}
            tokens = self._annotate_tokens(text)
            entities = _entities(annotated.answers)
            for entity in entities:
                if not entity.needs_masking:
                    continue
                found = self._is_found(text, tokens, masked, entity)
                if entity.is_direct:
                    direct_total += 1
                    direct_found += found
                else:
                    quasi_total += 1
                    quasi_found += found
            hits, masks = self._weighted_precision_counts(text, predicted, entities)
            weighted_hits += hits
            weighted_masks += masks
        return TabOfficialMetrics(
            er_di=direct_found / direct_total if direct_total else 0.0,
            er_qi=quasi_found / quasi_total if quasi_total else 0.0,
            wp=weighted_hits / weighted_masks if weighted_masks else 0.0,
        )

    def _is_found(self, text: str, tokens: Sequence[TokenInfo], masked: set[int], entity: _Entity) -> bool:
        for (start, end), needs_masking in zip(entity.mentions, entity.masking):
            if needs_masking and not self._is_mention_masked(text, tokens, masked, start, end):
                return False
        return True

    def _is_mention_masked(
        self, text: str, tokens: Sequence[TokenInfo], masked: set[int], start: int, end: int
    ) -> bool:
        uncovered = set(range(start, end)) - masked
        if uncovered:
            for token in tokens:
                if token.end > start and token.start < end and (
                    token.part_of_speech in IGNORED_PARTS_OF_SPEECH or token.lower in IGNORED_WORDS
                ):
                    uncovered -= set(range(token.start, token.end))
        return all(text[offset] in IGNORED_CHARACTERS for offset in uncovered)

    def _weighted_precision_counts(
        self, text: str, predicted: Sequence[Span], entities: Sequence[_Entity]
    ) -> tuple[float, float]:
        mask_tokens = [
            (span.start + match.start(), span.start + match.end())
            for span in predicted
            for match in _WORD.finditer(text[span.start : span.end])
        ]
        if not mask_tokens:
            return 0.0, 0.0
        annotator_count = len({entity.annotator for entity in entities})
        masking_mentions = [
            (start, end, entity.annotator)
            for entity in entities
            if entity.needs_masking
            for (start, end), masking in zip(entity.mentions, entity.masking)
            if masking
        ]
        weights = self._weigh_tokens(text, mask_tokens)
        hits = masks = 0.0
        for (start, end), weight in zip(mask_tokens, weights):
            covering = {a for s, e, a in masking_mentions if s <= start and e >= end}
            hits += len(covering) * weight
            masks += annotator_count * weight
        return hits, masks
