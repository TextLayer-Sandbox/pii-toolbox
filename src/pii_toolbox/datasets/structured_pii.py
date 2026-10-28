"""Loaders for two public synthetic datasets of structured PII, read from Hugging Face parquet files.

`nvidia/Nemotron-PII` and `gretelai/synthetic_pii_finance_multilingual` are each pinned to a
commit and their test split is downloaded into `structured/` under the data directory on first
use. Both mark character spans with a label and an exclusive end offset. Nemotron stores its
spans as a Python-literal list whose numbers may arrive as text; Gretel stores them as JSON.
Spans whose offsets cannot be read as whole numbers are dropped.
"""

import ast
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pii_toolbox.datasets.download import DatasetFile, default_data_dir, download_datasets, network_fetch

NEMOTRON_REPOSITORY = "huggingface.co/datasets/nvidia/Nemotron-PII"
NEMOTRON_REVISION = "b70ffaf5ff39e079776134c5bf4381f00a9fd1ed"
GRETEL_REPOSITORY = "huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual"
GRETEL_REVISION = "7b844d16738527a04264f50214cb426a4cea0897"

NEMOTRON_TEST_FILE = Path("structured") / "nemotron" / "test.parquet"
GRETEL_TEST_FILE = Path("structured") / "gretel" / "test.parquet"

_BATCH_SIZE = 1000

STRUCTURED_DATASET_FILES: list[DatasetFile] = [
    DatasetFile(
        f"https://{NEMOTRON_REPOSITORY}/resolve/{NEMOTRON_REVISION}/data/test-00000-of-00001.parquet",
        NEMOTRON_TEST_FILE.as_posix(),
    ),
    DatasetFile(
        f"https://{GRETEL_REPOSITORY}/resolve/{GRETEL_REVISION}/data/test-00000-of-00001.parquet",
        GRETEL_TEST_FILE.as_posix(),
    ),
]


@dataclass(frozen=True)
class LabelledSpan:
    """A labelled character range of a document; `end` is exclusive."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class StructuredDocument:
    """One synthetic document with its labelled spans; `language` is the dataset's own language or locale name."""

    text: str
    language: str
    spans: tuple[LabelledSpan, ...]


def download_structured_datasets(data_dir: Path | None = None) -> None:
    """Fetches the two test files not yet present under the data directory."""
    download_datasets(data_dir or default_data_dir(), network_fetch, STRUCTURED_DATASET_FILES)


def _offset(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _spans(entries: object) -> tuple[LabelledSpan, ...]:
    spans = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
            continue
        start, end = _offset(entry.get("start")), _offset(entry.get("end"))
        if start is None or end is None or start >= end:
            continue
        spans.append(LabelledSpan(start, end, entry["label"]))
    return tuple(spans)


def parse_nemotron_spans(raw: str) -> tuple[LabelledSpan, ...]:
    """Reads a Nemotron `spans` value, a Python-literal list of dicts with start, end and label."""
    try:
        return _spans(ast.literal_eval(raw))
    except (ValueError, SyntaxError):
        return ()


def parse_gretel_spans(raw: str) -> tuple[LabelledSpan, ...]:
    """Reads a Gretel `pii_spans` value, a JSON list of dicts with start, end and label."""
    try:
        return _spans(json.loads(raw))
    except ValueError:
        return ()


def _batches(path: Path, columns: list[str]) -> Iterator[dict[str, list]]:
    import pyarrow.parquet as parquet

    for batch in parquet.ParquetFile(path).iter_batches(batch_size=_BATCH_SIZE, columns=columns):
        yield batch.to_pydict()


def load_nemotron_test(data_dir: Path | None = None) -> Iterator[StructuredDocument]:
    """Yields the Nemotron-PII test documents; the language is the locale, `us` or `intl`."""
    path = (data_dir or default_data_dir()) / NEMOTRON_TEST_FILE
    for batch in _batches(path, ["text", "spans", "locale"]):
        for text, spans, locale in zip(batch["text"], batch["spans"], batch["locale"]):
            yield StructuredDocument(text, locale, parse_nemotron_spans(spans))


def load_gretel_test(data_dir: Path | None = None) -> Iterator[StructuredDocument]:
    """Yields the Gretel finance test documents; the language is the dataset's language name."""
    path = (data_dir or default_data_dir()) / GRETEL_TEST_FILE
    for batch in _batches(path, ["generated_text", "pii_spans", "language"]):
        for text, spans, language in zip(batch["generated_text"], batch["pii_spans"], batch["language"]):
            yield StructuredDocument(text, language, parse_gretel_spans(spans))
