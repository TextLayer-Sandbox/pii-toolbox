import subprocess
import sys
from pathlib import Path

from pii_toolbox.evaluation.model_manifest import MODEL_MANIFEST, default_model_dir, download_models

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "download_models.py"


def recording_fetcher(calls):
    def fetch(entry, target):
        calls.append(entry.name)
        (target / "weights.bin").write_text("synthetic")

    return fetch


def test_fetches_each_manifest_entry_once_inside_dest(tmp_path):
    dest = tmp_path / "models"
    calls = []
    download_models(dest, recording_fetcher(calls))
    assert sorted(calls) == sorted(
        [
            "urchade/gliner_multi_pii-v1",
            "Jean-Baptiste/roberta-large-ner-english",
            "en_core_web_lg",
            "en_core_web_sm",
            "sapienzanlp/maverick-mes-ontonotes",
            "nltk_punkt",
            "microsoft/deberta-v3-large",
            "microsoft/mdeberta-v3-base",
        ]
    )
    assert len(calls) == len(MODEL_MANIFEST)
    assert [p for p in tmp_path.iterdir()] == [dest]


def test_second_run_fetches_nothing(tmp_path):
    calls = []
    download_models(tmp_path, recording_fetcher(calls))
    calls.clear()
    assert download_models(tmp_path, recording_fetcher(calls)) == []
    assert calls == []


def test_failed_fetch_is_retried(tmp_path):
    def failing(entry, target):
        raise RuntimeError("network down")

    try:
        download_models(tmp_path, failing)
    except RuntimeError:
        pass
    calls = []
    download_models(tmp_path, recording_fetcher(calls))
    assert len(calls) == len(MODEL_MANIFEST)


def test_default_dir_follows_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PII_MODEL_DIR", str(tmp_path))
    assert default_model_dir() == tmp_path


def test_script_help_exits_zero():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True)
    assert result.returncode == 0


def test_punkt_and_deberta_are_fetched_once_into_their_folders(tmp_path):
    dest = tmp_path / "models"
    calls = []
    download_models(dest, recording_fetcher(calls))
    assert calls.count("nltk_punkt") == 1
    assert calls.count("microsoft/deberta-v3-large") == 1
    assert (dest / "nltk_punkt").is_dir()
    assert (dest / "huggingface_cache").is_dir()
    assert [p for p in tmp_path.iterdir()] == [dest]


def test_mdeberta_is_fetched_once_into_the_shared_huggingface_cache(tmp_path):
    dest = tmp_path / "models"
    calls = []
    download_models(dest, recording_fetcher(calls))
    assert calls.count("microsoft/mdeberta-v3-base") == 1
    assert calls.count("microsoft/deberta-v3-large") == 1
    assert (dest / "huggingface_cache").is_dir()
    assert [p for p in tmp_path.iterdir()] == [dest]


def test_refs_main_holds_the_pinned_revision_for_each_huggingface_cache_entry(tmp_path):
    dest = tmp_path / "models"
    download_models(dest, recording_fetcher([]))
    cached = [entry for entry in MODEL_MANIFEST if entry.source == "huggingface_cache"]
    assert len(cached) >= 2
    for entry in cached:
        organisation, name = entry.location.split("/")
        refs_main = dest / "huggingface_cache" / f"models--{organisation}--{name}" / "refs" / "main"
        assert refs_main.read_text() == entry.revision
