"""Loader for Worldwide English NER, read from the data directory filled by the download script.

Expects the pinned repository tarball at `worldwide/en-worldwide-newswire.tar.gz` under the
data directory. Each `processed_annotated/*.txt.tsv` member is one article, one token and
BIO tag per line, a blank line ending a sentence. Every article is used whatever the split,
because nothing is tuned on them. Tokens are joined by a space and sentences by a newline,
and answer spans are rebuilt from the BIO tags as character offsets in that text.
"""

import tarfile
from collections.abc import Iterator
from pathlib import Path

from pii_toolbox.core import AnnotatedDocument, AnswerSpan, Dataset, DatasetSplit, Document
from pii_toolbox.core.label_maps import WORLDWIDE_LABELS
from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.datasets.mapa import spans_from_bio

_TARBALL = Path("worldwide") / "en-worldwide-newswire.tar.gz"
_ARTICLE_FOLDER = "processed_annotated/"
_ARTICLE_SUFFIX = ".txt.tsv"


class WorldwideDataset(Dataset):
    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir if data_dir is not None else default_data_dir()

    @property
    def name(self) -> str:
        return "Worldwide"

    @property
    def languages(self) -> frozenset[str]:
        return frozenset({"en"})

    def load(self, split: DatasetSplit, language: str | None = None) -> Iterator[AnnotatedDocument]:
        if language is not None and language != "en":
            return
        with tarfile.open(self._data_dir / _TARBALL, "r:gz") as archive:
            articles = sorted(
                (member for member in archive.getmembers() if _is_article(member)),
                key=lambda member: member.name,
            )
            for member in articles:
                content = archive.extractfile(member)
                assert content is not None
                file_name = member.name.rsplit("/", 1)[-1]
                yield _document(file_name.removesuffix(_ARTICLE_SUFFIX), content.read().decode("utf-8"))


def _is_article(member: tarfile.TarInfo) -> bool:
    return member.isfile() and _ARTICLE_FOLDER in member.name and member.name.endswith(_ARTICLE_SUFFIX)


def _sentences(content: str) -> list[tuple[list[str], list[str]]]:
    sentences: list[tuple[list[str], list[str]]] = []
    tokens: list[str] = []
    tags: list[str] = []
    for line in content.splitlines():
        if not line.strip():
            if tokens:
                sentences.append((tokens, tags))
            tokens, tags = [], []
            continue
        columns = line.split("\t")
        if len(columns) < 2:
            continue
        tokens.append(columns[0])
        tags.append(columns[1])
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def _document(article_id: str, content: str) -> AnnotatedDocument:
    lines = []
    answers = []
    offset = 0
    for tokens, tags in _sentences(content):
        for start, end, label in spans_from_bio(tokens, tags, offset):
            answers.append(AnswerSpan(start, end, WORLDWIDE_LABELS[label], label, None, None, None))
        line = " ".join(tokens)
        lines.append(line)
        offset += len(line) + 1
    document = Document(id=article_id, text="\n".join(lines), language="en")
    return AnnotatedDocument(document=document, answers=tuple(answers))
