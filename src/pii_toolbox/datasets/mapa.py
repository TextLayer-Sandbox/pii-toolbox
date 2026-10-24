"""Loader for the MAPA corpus, read from the data directory filled by the download script.

Expects `mapa/train.jsonl`, `mapa/validation.jsonl` and `mapa/test.jsonl` under the data
directory, one sentence per line with `language`, `file_name`, `sentence_number`, `tokens`
and `coarse_grained` BIO tags. DEV is MAPA's validation file. Sentences of one source file
form one document: tokens are joined by a space and sentences by a newline, and answer
spans are rebuilt from the BIO tags as character offsets in that text.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from pii_toolbox.core import AnnotatedDocument, AnswerSpan, Dataset, DatasetSplit, Document
from pii_toolbox.core.label_maps import MAPA_LABELS
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
    """Returns (start, end, label) character spans of tokens joined by single spaces.

    An I- tag continues the open span of the same label and otherwise starts a new one,
    so a dangling I- tag opens a span.
    """
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


class MapaDataset(Dataset):
    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir if data_dir is not None else default_data_dir()

    @property
    def name(self) -> str:
        return "MAPA"

    @property
    def languages(self) -> frozenset[str]:
        return MAPA_LANGUAGES

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
    lines = []
    answers = []
    offset = 0
    for sentence in sentences:
        tokens = sentence["tokens"]
        for start, end, label in spans_from_bio(tokens, sentence["coarse_grained"], offset):
            answers.append(AnswerSpan(start, end, MAPA_LABELS[label], label, None, None, None))
        line = " ".join(tokens)
        lines.append(line)
        offset += len(line) + 1
    document = Document(id=file_name, text="\n".join(lines), language=language)
    return AnnotatedDocument(document=document, answers=tuple(answers))
