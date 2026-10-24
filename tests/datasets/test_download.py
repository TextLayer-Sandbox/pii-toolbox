import json
import subprocess
from pathlib import Path

import pytest

from pii_toolbox.datasets.download import DATASET_FILES, default_data_dir, download_datasets

REPO_ROOT = Path(__file__).resolve().parents[2]
ONE_MEGABYTE = 1024 * 1024


def recording_fetcher(calls):
    def fetch(url, target):
        calls.append(url)
        target.write_text("synthetic")

    return fetch


def test_requests_tab_and_mapa_and_writes_only_under_dest(tmp_path):
    dest = tmp_path / "data"
    calls = []
    download_datasets(dest, recording_fetcher(calls))
    assert any("NorskRegnesentral/text-anonymization-benchmark" in url for url in calls)
    assert any("huggingface.co/datasets/joelniklaus/mapa" in url for url in calls)
    assert len(calls) == len(DATASET_FILES)
    assert list(tmp_path.iterdir()) == [dest]
    assert not list(dest.rglob("*.partial"))


def test_second_run_fetches_nothing(tmp_path):
    calls = []
    download_datasets(tmp_path, recording_fetcher(calls))
    calls.clear()
    assert download_datasets(tmp_path, recording_fetcher(calls)) == []
    assert calls == []


def test_interrupted_fetch_leaves_no_present_file(tmp_path):
    def failing(url, target):
        target.write_text("half")
        raise RuntimeError("connection lost")

    with pytest.raises(RuntimeError):
        download_datasets(tmp_path, failing)
    calls = []
    download_datasets(tmp_path, recording_fetcher(calls))
    assert len(calls) == len(DATASET_FILES)


def test_default_dir_follows_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PII_DATA_DIR", str(tmp_path))
    assert default_data_dir() == tmp_path


def test_repository_tracks_no_dataset_files():
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout.decode()
    offenders = []
    for name in filter(None, listing.split("\0")):
        path = REPO_ROOT / name
        base = path.name.lower()
        too_big = path.is_file() and path.stat().st_size > ONE_MEGABYTE
        dataset_name = (
            (base.startswith("tab") and base.endswith(".json"))
            or base.startswith("mapa")
            or base.endswith(".parquet")
        )
        if too_big or dataset_name:
            offenders.append(name)
    assert offenders == []


@pytest.mark.data
def test_cache_holds_tab_splits_and_mapa_languages():
    root = default_data_dir()
    for split in ("train", "dev", "test"):
        assert (root / "tab" / f"echr_{split}.json").stat().st_size > 0
    for split in ("train", "validation", "test"):
        languages = set()
        with open(root / "mapa" / f"{split}.jsonl") as handle:
            for line in handle:
                languages.add(json.loads(line)["language"])
        assert {"en", "fr", "de"} <= languages
