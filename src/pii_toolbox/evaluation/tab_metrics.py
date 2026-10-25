"""TAB's own anonymisation metrics: entity recall on direct and quasi identifiers, and weighted precision.

The definitions follow the TAB benchmark. An entity is one annotator's set of mentions that
share an entity id. It needs masking when any of its mentions is marked DIRECT or QUASI, and
it is direct when its first mention is DIRECT. An entity counts as found only when every
mention that needs masking is covered by the predicted spans, ignoring punctuation, spaces,
function words and titles. Recall pools entities over annotators. Precision is taken over the
word tokens inside predicted spans: each token earns the share of annotators whose masking
mention fully covers it, weighted by how much information the token carries.
"""

import math
import re
import zipfile
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from intervaltree import Interval, IntervalTree

from pii_toolbox.core import (
    AnnotatedDocument,
    AnswerSpan,
    ReidentificationRisk,
    Span,
    TabMetricsScorer,
    TabOfficialMetrics,
)
from pii_toolbox.evaluation.model_manifest import default_model_dir

IGNORED_PARTS_OF_SPEECH = frozenset({"ADP", "PART", "CCONJ", "DET"})
IGNORED_WORDS = frozenset({"mr", "mrs", "ms", "no", "nr", "about"})
IGNORED_CHARACTERS = frozenset(" ,.-;:/&()[]–'\" ’“”")
_WORD = re.compile(r"\w+")
_MASKING_RISKS = frozenset({ReidentificationRisk.DIRECT, ReidentificationRisk.QUASI})
SPACY_MODEL_NAME = "en_core_web_md"
BERT_MODEL_LOCATION = "google-bert/bert-base-uncased"
BERT_SEGMENT_SIZE = 100


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
            tokens = list(self._annotate_tokens(text))
            token_ends = [token.end for token in tokens]
            entities = _entities(annotated.answers)
            for entity in entities:
                if not entity.needs_masking:
                    continue
                found = self._is_found(text, tokens, token_ends, masked, entity)
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

    def _is_found(
        self, text: str, tokens: Sequence[TokenInfo], token_ends: Sequence[int], masked: set[int], entity: _Entity
    ) -> bool:
        for (start, end), needs_masking in zip(entity.mentions, entity.masking):
            if needs_masking and not self._is_mention_masked(text, tokens, token_ends, masked, start, end):
                return False
        return True

    def _is_mention_masked(
        self,
        text: str,
        tokens: Sequence[TokenInfo],
        token_ends: Sequence[int],
        masked: set[int],
        start: int,
        end: int,
    ) -> bool:
        uncovered = set(range(start, end)) - masked
        if uncovered:
            for token in tokens[bisect_right(token_ends, start) :]:
                if token.start >= end:
                    break
                if token.part_of_speech in IGNORED_PARTS_OF_SPEECH or token.lower in IGNORED_WORDS:
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


def spacy_token_annotator(model_dir: Path | None = None) -> TokenAnnotator:
    """Tags tokens with the English medium spaCy model unpacked from its wheel in the model directory."""
    import spacy

    wheel_folder = (model_dir or default_model_dir()) / SPACY_MODEL_NAME
    unpacked = wheel_folder / "unpacked"
    if not unpacked.exists():
        wheels = sorted(wheel_folder.glob("*.whl"))
        if not wheels:
            raise FileNotFoundError(f"no wheel for {SPACY_MODEL_NAME} in {wheel_folder}; run scripts/download_models.py")
        with zipfile.ZipFile(wheels[0]) as archive:
            archive.extractall(unpacked)
    pipeline = spacy.load(next(unpacked.glob(f"{SPACY_MODEL_NAME}/{SPACY_MODEL_NAME}-*")))

    def annotate(text: str) -> list[TokenInfo]:
        return [TokenInfo(token.idx, token.idx + len(token), token.pos_, token.lower_) for token in pipeline(text)]

    return annotate


class BertInformationWeigher:
    """Weighs each span by how surprising it is to BERT: the negative log probability of its tokens when masked.

    A span covering several BERT tokens takes the lowest token probability; a span covering
    none weighs zero. Long texts are cut into fixed-size segments so each fits the model.
    The model loads from the local Hugging Face cache only.
    """

    def __init__(self, model_dir: Path | None = None, segment_size: int = BERT_SEGMENT_SIZE) -> None:
        import torch
        from transformers import BertForMaskedLM, BertTokenizerFast

        cache_dir = str((model_dir or default_model_dir()) / "huggingface_cache")
        self._tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_LOCATION, cache_dir=cache_dir, local_files_only=True)
        self._model = BertForMaskedLM.from_pretrained(BERT_MODEL_LOCATION, cache_dir=cache_dir, local_files_only=True)
        self._model.eval()
        self._segment_size = segment_size
        self._torch = torch

    def __call__(self, text: str, spans: Sequence[tuple[int, int]]) -> list[float]:
        encoding = self._tokenizer(text, return_offsets_mapping=True)
        input_ids = list(encoding["input_ids"])
        original_ids = list(input_ids)
        attention = list(encoding["attention_mask"])
        span_tree = IntervalTree(Interval(start, end) for start, end in set(spans) if start < end)
        tokens_by_span: dict[tuple[int, int], list[int]] = {span: [] for span in spans}
        for index, (token_start, token_end) in enumerate(encoding["offset_mapping"]):
            for overlapping in span_tree.overlap(token_start, token_end):
                tokens_by_span[(overlapping.begin, overlapping.end)].append(index)
        for indices in tokens_by_span.values():
            for index in indices:
                input_ids[index] = self._tokenizer.mask_token_id
                attention[index] = 0
        probabilities = self._token_probabilities(input_ids, attention, original_ids)
        return [
            -math.log(min(probabilities[index] for index in tokens_by_span[span])) if tokens_by_span[span] else 0.0
            for span in spans
        ]

    def _token_probabilities(self, input_ids: list[int], attention: list[int], original_ids: list[int]) -> list[float]:
        torch = self._torch
        ids = torch.tensor(input_ids)
        mask = torch.tensor(attention)
        split_at = list(range(self._segment_size, len(input_ids), self._segment_size))
        id_segments = torch.tensor_split(ids, split_at)
        mask_segments = torch.tensor_split(mask, split_at)
        padded_ids = torch.nn.utils.rnn.pad_sequence(id_segments, batch_first=True)
        padded_mask = torch.nn.utils.rnn.pad_sequence(mask_segments, batch_first=True)
        with torch.no_grad():
            logits = self._model(input_ids=padded_ids, attention_mask=padded_mask).logits
        logits = torch.vstack(list(logits))[: len(input_ids)]
        probabilities = torch.softmax(logits, dim=1)
        return probabilities[torch.arange(len(input_ids)), torch.tensor(original_ids)].tolist()
