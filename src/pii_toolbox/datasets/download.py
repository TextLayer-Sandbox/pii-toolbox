"""Downloads the TAB, MAPA and Worldwide English NER corpora into a data directory outside the repository."""

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
WORLDWIDE_REPOSITORY = "github.com/stanfordnlp/en-worldwide-newswire"
WORLDWIDE_REVISION = "53cee6af7a1eec472db398de2d2fdcd2a032dcc9"


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


def _worldwide_tarball() -> DatasetFile:
    url = f"https://codeload.{WORLDWIDE_REPOSITORY}/tar.gz/{WORLDWIDE_REVISION}"
    return DatasetFile(url, "worldwide/en-worldwide-newswire.tar.gz")


DATASET_FILES: list[DatasetFile] = [
    _tab_file("echr_train.json"),
    _tab_file("echr_dev.json"),
    _tab_file("echr_test.json"),
    _tab_file("evaluation.py"),
    _mapa_file("train.jsonl"),
    _mapa_file("validation.jsonl"),
    _mapa_file("test.jsonl"),
    _worldwide_tarball(),
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
