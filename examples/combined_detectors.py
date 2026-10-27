"""Runnable combination: a name detector plus Presidio's structured recognisers in one union.

Needs the downloaded models (`uv run python scripts/download_models.py`).
Run from the repository root with `uv run python examples/combined_detectors.py`.
"""

from pii_toolbox.core import Document
from pii_toolbox.detection.registry import build_from_config

TEXT = "Maria Example can be reached at maria.example@example.com or +1 212 555 0142."


def main() -> None:
    document = Document("example", TEXT, "en")
    # Names alone miss the structured values, and the structured recognisers alone miss the name.
    for entry in ["spacy_lg", "presidio_structured", "spacy_lg+presidio_structured"]:
        (detector,) = build_from_config([entry])
        detector.load_models()
        print(f"{entry}:")
        for span in detector.detect(document):
            print(f"  {span.pii_type.name:<10} {span.detector_label:<14} {TEXT[span.start : span.end]!r}")


if __name__ == "__main__":
    main()
