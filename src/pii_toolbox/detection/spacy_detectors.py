"""Detectors that read the named entities found by an installed-from-wheel spaCy model.

The model is unpacked from the wheel kept in the model directory and loaded from disk,
so nothing is downloaded or imported from site-packages at detection time.
"""

import shutil
import zipfile
from pathlib import Path
from typing import Any

from pii_toolbox.core import Detector, Document, PiiType, Span
from pii_toolbox.evaluation.model_manifest import default_model_dir

SPACY_LABELS: dict[str, PiiType] = {
    "PERSON": PiiType.PERSON,
    "ORG": PiiType.ORGANIZATION,
    "GPE": PiiType.LOCATION,
    "LOC": PiiType.LOCATION,
    "DATE": PiiType.DATE_TIME,
    "TIME": PiiType.DATE_TIME,
    "MONEY": PiiType.QUANTITY,
    "QUANTITY": PiiType.QUANTITY,
}

_UNPACKED_FOLDER = "unpacked"


def unpack_model(model_dir: Path, model_name: str) -> Path:
    """Extracts the model wheel once and returns the folder spaCy loads from."""
    wheel_folder = model_dir / model_name
    unpacked = wheel_folder / _UNPACKED_FOLDER
    if not unpacked.exists():
        wheels = sorted(wheel_folder.glob("*.whl"))
        if not wheels:
            raise FileNotFoundError(f"no wheel for {model_name} in {wheel_folder}; run scripts/download_models.py")
        partial = wheel_folder / f"{_UNPACKED_FOLDER}.partial"
        shutil.rmtree(partial, ignore_errors=True)
        with zipfile.ZipFile(wheels[0]) as archive:
            archive.extractall(partial)
        partial.rename(unpacked)
    return next(unpacked.glob(f"{model_name}/{model_name}-*"))


class SpacyDetector(Detector):
    """Reports the person, organisation, place, date, time and amount entities of a spaCy pipeline.

    `pipeline` may be given to supply an already built pipeline instead of loading the model.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        model_dir: Path | None = None,
        pipeline: Any = None,
    ) -> None:
        self._name = name
        self._model_name = model_name
        self._model_dir = model_dir
        self._pipeline = pipeline

    @property
    def name(self) -> str:
        return self._name

    @property
    def languages(self) -> frozenset[str] | None:
        return frozenset({"en"})

    @property
    def detects(self) -> frozenset[PiiType]:
        return frozenset(SPACY_LABELS.values())

    def load_models(self) -> None:
        import spacy

        model_dir = self._model_dir or default_model_dir()
        self._pipeline = spacy.load(unpack_model(model_dir, self._model_name))

    def detect(self, document: Document) -> list[Span]:
        if self._pipeline is None:
            self.load_models()
        return [
            Span(
                start=entity.start_char,
                end=entity.end_char,
                pii_type=SPACY_LABELS[entity.label_],
                detector_label=entity.label_,
                source=(self._name,),
                score=None,
            )
            for entity in self._pipeline(document.text).ents
            if entity.label_ in SPACY_LABELS and any(c.isalnum() for c in document.text[entity.start_char : entity.end_char])
        ]


def spacy_lg_detector(model_dir: Path | None = None) -> SpacyDetector:
    return SpacyDetector("spacy_lg", "en_core_web_lg", model_dir)


def spacy_sm_detector(model_dir: Path | None = None) -> SpacyDetector:
    return SpacyDetector("spacy_sm", "en_core_web_sm", model_dir)
