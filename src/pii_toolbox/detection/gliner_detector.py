"""Detector running the multilingual GLiNER PII model with a fixed list of requested labels.

GLiNER reads a limited number of tokens, so long text is cut into overlapping segments.
"""

from pathlib import Path
from typing import Any

from pii_toolbox.core import GLINER_LABELS, Detector, Document, PiiType, Span, to_pii_type
from pii_toolbox.detection.segmenter import OverlappingTextSegmenter, detect_segmented
from pii_toolbox.evaluation.model_manifest import default_model_dir

MODEL_FOLDER = "urchade__gliner_multi_pii-v1"
REQUESTED_LABELS = ("person", "organization", "location", "address")
SEGMENT_CHARACTERS = 1200
SEGMENT_OVERLAP_CHARACTERS = 150
SCORE_THRESHOLD = 0.5


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
        return frozenset({PiiType.PERSON, PiiType.ORGANIZATION, PiiType.LOCATION})

    def load_models(self) -> None:
        from gliner import GLiNER

        path = (self._model_dir or default_model_dir()) / MODEL_FOLDER
        if not path.exists():
            raise FileNotFoundError(f"model folder missing: {path}; run scripts/download_models.py")
        self._model = GLiNER.from_pretrained(str(path), local_files_only=True)

    def detect(self, document: Document) -> list[Span]:
        if self._model is None:
            self.load_models()
        return detect_segmented(document.text, self._segmenter, self._detect_segment)

    def _detect_segment(self, text: str) -> list[Span]:
        spans = []
        for entity in self._model.predict_entities(text, list(REQUESTED_LABELS), threshold=SCORE_THRESHOLD):
            if entity["label"] not in REQUESTED_LABELS or entity["start"] >= entity["end"]:
                continue
            spans.append(
                Span(
                    start=entity["start"],
                    end=entity["end"],
                    pii_type=to_pii_type(GLINER_LABELS, entity["label"]),
                    detector_label=entity["label"],
                    source=(self.name,),
                    score=float(entity["score"]),
                )
            )
        return spans
