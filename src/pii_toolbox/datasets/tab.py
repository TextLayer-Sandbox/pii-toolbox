"""Loader for the TAB corpus, read from the data directory filled by the download script.

Expects `tab/echr_train.json`, `tab/echr_dev.json` and `tab/echr_test.json` under the data
directory. Each file is a list of documents; each document holds an `annotations` object
keyed by annotator, and each annotator lists entity mentions with character offsets, a
TAB label, an identifier type (DIRECT, QUASI or NO_MASK) and an entity id. Every mention
of every annotator becomes its own answer span, so annotators with different boundaries
stay separate. All TAB documents are English.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from pii_toolbox.core import (
    AnnotatedDocument,
    AnswerSpan,
    Dataset,
    DatasetSplit,
    Document,
    ReidentificationRisk,
)
from pii_toolbox.core.label_maps import TAB_LABELS
from pii_toolbox.datasets.download import default_data_dir

LANGUAGE = "en"

_SPLIT_FILES = {
    DatasetSplit.TRAIN: "echr_train.json",
    DatasetSplit.DEV: "echr_dev.json",
    DatasetSplit.TEST: "echr_test.json",
}


_RISKS = {
    "DIRECT": ReidentificationRisk.DIRECT,
    "QUASI": ReidentificationRisk.QUASI,
    "NO_MASK": ReidentificationRisk.NONE,
}


class TabDataset(Dataset):
    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir if data_dir is not None else default_data_dir()

    @property
    def name(self) -> str:
        return "TAB"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({LANGUAGE})

    def load(self, split: DatasetSplit, language: str | None = None) -> Iterator[AnnotatedDocument]:
        if language not in (None, LANGUAGE):
            return
        with open(self._data_dir / "tab" / _SPLIT_FILES[split], encoding="utf-8") as handle:
            raw_documents = json.load(handle)
        for raw in raw_documents:
            document = Document(id=raw["doc_id"], text=raw["text"], language=LANGUAGE)
            yield AnnotatedDocument(document=document, answers=tuple(_answers(raw["annotations"])))


def _answers(annotations: dict) -> Iterator[AnswerSpan]:
    for annotator, content in annotations.items():
        for mention in content["entity_mentions"]:
            label = mention["entity_type"]
            yield AnswerSpan(
                start=mention["start_offset"],
                end=mention["end_offset"],
                pii_type=TAB_LABELS[label],
                label=label,
                annotator=annotator,
                risk=_RISKS[mention["identifier_type"]],
                entity_id=mention.get("entity_id"),
            )
