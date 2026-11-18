"""Loader for the MAPA corpus, read from the data directory."""

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from pii_toolbox.core import AnnotatedDocument, AnswerSpan, Dataset, DatasetSplit, Document, PiiType
from pii_toolbox.core.label_maps import MAPA_LABELS, OptionalLabelMap
from pii_toolbox.datasets.download import default_data_dir

MAPA_LANGUAGES = frozenset(
    "bg cs da de el en es et fi fr ga hu it lt lv mt nl pt ro sk sv".split()
)

_SPLIT_FILES = {
    DatasetSplit.TRAIN: "train.jsonl",
    DatasetSplit.DEV: "validation.jsonl",
    DatasetSplit.TEST: "test.jsonl",
}


def spans_from_bio(tokens: Sequence[str], tags: Sequence[str], offset: int = 0) -> list[tuple[int, int, str]]:
    """Returns (start, end, label) character spans of tokens joined by single spaces, where an I- tag continues an open span of the same label and otherwise starts a new one."""
    positions = []
    position = offset
    for token in tokens:
        positions.append((position, position + len(token)))
        position += len(token) + 1

    spans: list[tuple[int, int, str]] = []
    open_label: str | None = None
    open_start = 0
    open_end = 0
    for (start, end), tag in zip(positions, tags):
        prefix, _, label = tag.partition("-")
        if prefix == "I" and label == open_label:
            open_end = end
            continue
        if open_label is not None:
            spans.append((open_start, open_end, open_label))
            open_label = None
        if prefix in ("B", "I"):
            open_label, open_start, open_end = label, start, end
    if open_label is not None:
        spans.append((open_start, open_end, open_label))
    return spans


def document_from_sentences(
    document_id: str,
    language: str,
    sentences: Iterable[tuple[Sequence[str], Sequence[str]]],
    labels: OptionalLabelMap,
) -> AnnotatedDocument:
    """Builds one document from (tokens, BIO tags) sentences, joining tokens by a space and sentences by a newline, with answer spans typed through `labels`."""
    lines = []
    answers = []
    offset = 0
    for tokens, tags in sentences:
        for start, end, label in spans_from_bio(tokens, tags, offset):
            answers.append(AnswerSpan(start, end, labels[label], label, None, None, None))
        line = " ".join(tokens)
        lines.append(line)
        offset += len(line) + 1
    document = Document(id=document_id, text="\n".join(lines), language=language)
    return AnnotatedDocument(document=document, answers=tuple(answers))


class MapaDataset(Dataset):
    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir if data_dir is not None else default_data_dir()

    @property
    def name(self) -> str:
        return "MAPA"

    @property
    def languages(self) -> frozenset[str]:
        return MAPA_LANGUAGES

    @property
    def annotated_types(self) -> frozenset[PiiType]:
        return frozenset(PiiType) - {PiiType.IDENTIFIER}

    def load(self, split: DatasetSplit, language: str | None = None) -> Iterator[AnnotatedDocument]:
        sentences_by_file: dict[tuple[str, str], list[dict]] = {}
        with open(self._data_dir / "mapa" / _SPLIT_FILES[split], encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if language is not None and record["language"] != language:
                    continue
                sentences_by_file.setdefault((record["language"], record["file_name"]), []).append(record)
        for (file_language, file_name), sentences in sentences_by_file.items():
            sentences.sort(key=lambda sentence: sentence["sentence_number"])
            yield _document(file_language, file_name, sentences)


def _document(language: str, file_name: str, sentences: list[dict]) -> AnnotatedDocument:
    return document_from_sentences(
        file_name,
        language,
        ((sentence["tokens"], sentence["coarse_grained"]) for sentence in sentences),
        MAPA_LABELS,
    )
