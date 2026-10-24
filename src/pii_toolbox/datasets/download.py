"""Downloads the TAB and MAPA corpora into a data directory outside the repository.

The directory is PII_DATA_DIR, defaulting to a folder under the user cache. Layout under it:

    tab/echr_train.json  tab/echr_dev.json  tab/echr_test.json  tab/evaluation.py
    mapa/train.jsonl     mapa/validation.jsonl     mapa/test.jsonl

TAB comes from github.com/NorskRegnesentral/text-anonymization-benchmark and MAPA from
huggingface.co/datasets/joelniklaus/mapa, each pinned to a commit. Each MAPA file holds
every language, one JSON record per line with a "language" field (en, fr, de among them).
A file is fetched to a temporary name and renamed on success, so an interrupted run
leaves no partial file that later counts as present.
"""

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DATA_DIR_ENV_VAR = "PII_DATA_DIR"

TAB_REPOSITORY = "github.com/NorskRegnesentral/text-anonymization-benchmark"
TAB_REVISION = "558e09e26d6b36f5f78440074e6a233946d98bd9"
MAPA_REPOSITORY = "huggingface.co/datasets/joelniklaus/mapa"
MAPA_REVISION = "bbb2a0157b760465002fd12a61af81b475cd387a"


@dataclass(frozen=True)
class DatasetFile:
    """One file to download: its source URL and its path relative to the data directory."""

    url: str
    relative_path: str


Fetcher = Callable[[str, Path], None]


def _tab_file(name: str) -> DatasetFile:
    url = f"https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/{TAB_REVISION}/{name}"
    return DatasetFile(url, f"tab/{name}")


def _mapa_file(name: str) -> DatasetFile:
    url = f"https://huggingface.co/datasets/joelniklaus/mapa/resolve/{MAPA_REVISION}/{name}"
    return DatasetFile(url, f"mapa/{name}")


DATASET_FILES: list[DatasetFile] = [
    _tab_file("echr_train.json"),
    _tab_file("echr_dev.json"),
    _tab_file("echr_test.json"),
    _tab_file("evaluation.py"),
    _mapa_file("train.jsonl"),
    _mapa_file("validation.jsonl"),
    _mapa_file("test.jsonl"),
]


def default_data_dir() -> Path:
    configured = os.environ.get(DATA_DIR_ENV_VAR)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "pii-toolbox" / "data"


def download_datasets(dest: Path, fetch: Fetcher, files: list[DatasetFile] | None = None) -> list[DatasetFile]:
    """Fetch every dataset file not yet present under dest; return the files fetched."""
    fetched = []
    for item in DATASET_FILES if files is None else files:
        target = dest / item.relative_path
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        fetch(item.url, partial)
        partial.rename(target)
        fetched.append(item)
    return fetched


def network_fetch(url: str, target: Path) -> None:
    """Download url into target."""
    urllib.request.urlretrieve(url, target)
