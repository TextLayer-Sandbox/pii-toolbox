"""Runnable co-reference comparison: mask one text without and with identity resolution.

Needs the downloaded models (`uv run python scripts/download_models.py`).
Run from the repository root with `uv run python examples/coreference.py`.
"""

from pii_toolbox.coref.resolver import MaverickResolver
from pii_toolbox.detection.registry import build_from_config
from pii_toolbox.session import AngleBracketFormat, InMemoryPlaceholderStore, Session

TEXT = (
    "Jonathan Smith joined Sample Corp in 2020. Jon enjoys the work, and Mr Smith "
    "was promoted last spring. Colleagues say Smith is easy to reach at jon.smith@example.com."
)


def mask_text(resolver) -> str:
    (detector,) = build_from_config(["spacy_lg+presidio_structured"])
    detector.load_models()
    with Session(detector, InMemoryPlaceholderStore(), AngleBracketFormat(), "en", resolver=resolver) as session:
        return session.mask(TEXT).masked_text


def main() -> None:
    print(f"original: {TEXT}")
    print(f"co-reference off: {mask_text(None)}")
    # The worker process is started and stopped by the context manager.
    with MaverickResolver() as resolver:
        print(f"co-reference on: {mask_text(resolver)}")


if __name__ == "__main__":
    main()
