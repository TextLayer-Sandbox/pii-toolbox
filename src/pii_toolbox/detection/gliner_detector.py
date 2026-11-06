"""Detector running the multilingual GLiNER PII model with a fixed list of requested labels.

GLiNER reads a limited number of tokens, so long text is cut into overlapping segments.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pii_toolbox.core import GLINER_LABELS, Detector, Document, PiiType, Span
from pii_toolbox.detection.segmenter import OverlappingTextSegmenter, detect_segmented
from pii_toolbox.evaluation.model_manifest import default_model_dir

MODEL_FOLDER = "urchade__gliner_multi_pii-v1"
REQUESTED_LABELS = (
    "person",
    "organization",
    "location",
    "address",
    "date",
    "money",
    "case number",
    "identification number",
)
SEGMENT_CHARACTERS = 1200
SEGMENT_OVERLAP_CHARACTERS = 150
SCORE_THRESHOLD = 0.5
HUGGINGFACE_CACHE_FOLDER = "huggingface_cache"


@contextmanager
def _offline_huggingface(cache_dir: Path) -> Iterator[None]:
    """Makes Hugging Face loaders read only `cache_dir`, because the GLiNER config names a backbone model it fetches by hub name."""
    from huggingface_hub import constants

    previous_env = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "HF_HUB_CACHE")}
    previous_constants = (constants.HF_HUB_OFFLINE, constants.HF_HUB_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    constants.HF_HUB_OFFLINE = True
    constants.HF_HUB_CACHE = str(cache_dir)
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE, constants.HF_HUB_CACHE = previous_constants
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class GlinerDetector(Detector):
    """Reports the entities GLiNER finds for the requested labels.

    `model` may be given to supply an already loaded model offering
    `predict_entities(text, labels, threshold)` that returns dicts with start, end, label and score.
    """

    def __init__(self, model_dir: Path | None = None, model: Any = None) -> None:
        self._model_dir = model_dir
        self._model = model
        self._segmenter = OverlappingTextSegmenter(SEGMENT_CHARACTERS, SEGMENT_OVERLAP_CHARACTERS)

    @property
    def name(self) -> str:
        return "gliner"

    @property
    def languages(self) -> frozenset[str] | None:
        return None

    @property
    def detects(self) -> frozenset[PiiType]:
        return frozenset(
            {
                PiiType.PERSON,
                PiiType.ORGANIZATION,
                PiiType.LOCATION,
                PiiType.DATE_TIME,
                PiiType.QUANTITY,
                PiiType.IDENTIFIER,
            }
        )

    def load_models(self) -> None:
        from gliner import GLiNER

        model_dir = self._model_dir or default_model_dir()
        path = model_dir / MODEL_FOLDER
        if not path.exists():
            raise FileNotFoundError(f"model folder missing: {path}; run scripts/download_models.py")
        with _offline_huggingface(model_dir / HUGGINGFACE_CACHE_FOLDER):
            self._model = GLiNER.from_pretrained(str(path), local_files_only=True)

    def detect(self, document: Document) -> list[Span]:
        if self._model is None:
            self.load_models()
        return detect_segmented(document.text, self._segmenter, self._detect_segment)

    def _detect_segment(self, text: str) -> list[Span]:
        spans = []
        for entity in self._model.predict_entities(text, list(REQUESTED_LABELS), threshold=SCORE_THRESHOLD):
            if entity["label"] not in REQUESTED_LABELS or not any(c.isalnum() for c in text[entity["start"] : entity["end"]]):
                continue
            spans.append(
                Span(
                    start=entity["start"],
                    end=entity["end"],
                    pii_type=GLINER_LABELS[entity["label"]],
                    detector_label=entity["label"],
                    source=(self.name,),
                    score=float(entity["score"]),
                )
            )
        return spans
