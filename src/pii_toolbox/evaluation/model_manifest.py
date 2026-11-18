"""Manifest of every model the toolbox loads, and the download logic that fills a model directory."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

COMPLETE_MARKER_PREFIX = ".complete-"


@dataclass(frozen=True)
class ModelEntry:
    """One downloadable model: a source kind, where it comes from and the pinned revision."""

    name: str
    source: str
    revision: str
    location: str
    # Folder name under the model directory; when None the folder is derived from the model name.
    directory: str | None = None

    @property
    def marker_name(self) -> str:
        """Per-entry so entries sharing one folder are tracked separately."""
        return COMPLETE_MARKER_PREFIX + self.name.replace("/", "__")

    @property
    def directory_name(self) -> str:
        return self.directory or self.name.replace("/", "__")


Fetcher = Callable[[ModelEntry, Path], None]

MODEL_MANIFEST: list[ModelEntry] = [
    ModelEntry("urchade/gliner_multi_pii-v1", "huggingface", "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d", "urchade/gliner_multi_pii-v1"),
    ModelEntry("Jean-Baptiste/roberta-large-ner-english", "huggingface", "8f3abc1ef81ffbbb0e80568d4fed1dd10d459548", "Jean-Baptiste/roberta-large-ner-english"),
    ModelEntry("en_core_web_lg", "spacy", "3.8.0", "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"),
    ModelEntry("en_core_web_md", "spacy", "3.8.0", "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl"),
    ModelEntry("en_core_web_sm", "spacy", "3.8.0", "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"),
    ModelEntry("sapienzanlp/maverick-mes-ontonotes", "huggingface", "087a90f0380ec1f20ed19732b657f53fd3542c81", "sapienzanlp/maverick-mes-ontonotes"),
    ModelEntry("nltk_punkt", "download", "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a", "https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt.zip", directory="nltk_punkt"),
    ModelEntry("microsoft/deberta-v3-large", "huggingface_cache", "64a8c8eab3e352a784c658aef62be1662607476f", "microsoft/deberta-v3-large", directory="huggingface_cache"),
    ModelEntry("microsoft/mdeberta-v3-base", "huggingface_cache", "a0484667b22365f84929a935b5e50a51f71f159d", "microsoft/mdeberta-v3-base", directory="huggingface_cache"),
    ModelEntry("google-bert/bert-base-uncased", "huggingface_cache", "86b5e0934494bd15c9632b12f734a8a67f723594", "google-bert/bert-base-uncased", directory="huggingface_cache"),
]


def is_present(dest: Path, entry: ModelEntry) -> bool:
    return (dest / entry.directory_name / entry.marker_name).exists()


def write_hub_ref(cache_dir: Path, entry: ModelEntry) -> None:
    """Records the pinned revision as the cache's main ref so loading by hub name works offline."""
    organisation, name = entry.location.split("/")
    refs = cache_dir / f"models--{organisation}--{name}" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(entry.revision)


def download_models(dest: Path, fetch: Fetcher, manifest: list[ModelEntry] | None = None) -> list[ModelEntry]:
    """Fetches every manifest entry not yet present under dest, writing each completion marker only after its fetch returns, and returns the entries fetched."""
    fetched = []
    for entry in MODEL_MANIFEST if manifest is None else manifest:
        if is_present(dest, entry):
            continue
        target = dest / entry.directory_name
        target.mkdir(parents=True, exist_ok=True)
        fetch(entry, target)
        if entry.source == "huggingface_cache":
            write_hub_ref(target, entry)
        (target / entry.marker_name).write_text(entry.revision)
        fetched.append(entry)
    return fetched


def network_fetch(entry: ModelEntry, target: Path) -> None:
    """Download one entry from its real source into target."""
    if entry.source == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(entry.location, revision=entry.revision, local_dir=target)
        return
    if entry.source == "huggingface_cache":
        from huggingface_hub import snapshot_download

        snapshot_download(entry.location, revision=entry.revision, cache_dir=target)
        return
    if entry.source in ("spacy", "download"):
        import urllib.request

        urllib.request.urlretrieve(entry.location, target / entry.location.rsplit("/", 1)[-1])
        return
    raise ValueError(f"unknown model source: {entry.source}")
