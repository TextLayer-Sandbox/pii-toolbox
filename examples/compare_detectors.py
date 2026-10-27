"""Runnable comparison: run one text through each single detector and print what each finds.

Needs the downloaded models (`uv run python scripts/download_models.py`).
Run from the repository root with `uv run python examples/compare_detectors.py`.
"""

from pii_toolbox.core import Document
from pii_toolbox.detection.registry import build_from_config

TEXT = (
    "Maria Example of Sample Corp visited Springfield on 3 March 2024. "
    "Contact her at maria.example@example.com or +1 212 555 0142."
)
DETECTOR_NAMES = ["gliner", "roberta", "spacy_lg", "spacy_sm", "presidio_structured"]


def main() -> None:
    document = Document("example", TEXT, "en")
    print(f"text: {TEXT}")
    for name, detector in zip(DETECTOR_NAMES, build_from_config(DETECTOR_NAMES)):
        detector.load_models()
        print(f"\n{name}:")
        for span in detector.detect(document):
            print(f"  {span.pii_type.name:<12} {span.detector_label:<14} {TEXT[span.start : span.end]!r}")


if __name__ == "__main__":
    main()
