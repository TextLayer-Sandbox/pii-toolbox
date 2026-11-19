"""Detector running the RoBERTa English named-entity model through a token-classification pipeline."""

from pathlib import Path
from typing import Any

from pii_toolbox.core import ROBERTA_LABELS, Detector, Document, PiiType, Span
from pii_toolbox.detection.segmenter import OverlappingTextSegmenter, detect_segmented
from pii_toolbox.core.model_dir import default_model_dir

MODEL_FOLDER = "Jean-Baptiste__roberta-large-ner-english"
SEGMENT_CHARACTERS = 1500
SEGMENT_OVERLAP_CHARACTERS = 200


class RobertaDetector(Detector):
    """Reports person, organisation and location entities found by RoBERTa; an already built `pipeline` may be supplied."""

    def __init__(self, model_dir: Path | None = None, pipeline: Any = None) -> None:
        self._model_dir = model_dir
        self._pipeline: Any = pipeline
        self._segmenter = OverlappingTextSegmenter(SEGMENT_CHARACTERS, SEGMENT_OVERLAP_CHARACTERS)

    @property
    def name(self) -> str:
        return "roberta"

    @property
    def languages(self) -> frozenset[str] | None:
        return frozenset({"en"})

    @property
    def detects(self) -> frozenset[PiiType]:
        return frozenset({PiiType.PERSON, PiiType.ORGANIZATION, PiiType.LOCATION})

    def load_models(self) -> None:
        from transformers import pipeline

        path = (self._model_dir or default_model_dir()) / MODEL_FOLDER
        if not path.exists():
            raise FileNotFoundError(f"model folder missing: {path}; run scripts/download_models.py")
        self._pipeline = pipeline("token-classification", model=str(path), aggregation_strategy="simple")

    def detect(self, document: Document) -> list[Span]:
        if self._pipeline is None:
            self.load_models()
        return detect_segmented(document.text, self._segmenter, self._detect_segment)

    def _detect_segment(self, text: str) -> list[Span]:
        spans = []
        for entity in self._pipeline(text):
            pii_type = ROBERTA_LABELS[entity["entity_group"]]
            if pii_type is None or not any(c.isalnum() for c in text[entity["start"] : entity["end"]]):
                continue
            spans.append(
                Span(
                    start=entity["start"],
                    end=entity["end"],
                    pii_type=pii_type,
                    detector_label=entity["entity_group"],
                    source=(self.name,),
                    score=float(entity["score"]),
                )
            )
        return spans
