"""Runnable known-names example: always mask a list of names you already know about.

Run from the repository root with `uv run python examples/known_names.py`.
"""

from pii_toolbox.detection.known_names import KnownNamesDetector
from pii_toolbox.session import AngleBracketFormat, InMemoryPlaceholderStore, Session

# Staff of a made-up client, supplied up front so no model has to recognise them.
CLIENT_STAFF = ("Rosa Example", "Rose", "Tomas Sample")

TEXT = (
    "Rosa Example asked Tomas Sample to meet. rose is a flower, ROSE is shouting, "
    "and Rosemary is someone else."
)


def mask_with(detector: KnownNamesDetector) -> str:
    with Session(detector, InMemoryPlaceholderStore(), AngleBracketFormat(), "en") as session:
        return session.mask(TEXT).masked_text


def main() -> None:
    # By default a match must start with a capital letter, so ordinary words such as "rose" survive.
    print(f"default: {mask_with(KnownNamesDetector(CLIENT_STAFF))}")
    # Case-insensitive matching also catches names typed in lower case, at the cost of masking the flower.
    print(f"any case: {mask_with(KnownNamesDetector(CLIENT_STAFF, case_insensitive=True))}")


if __name__ == "__main__":
    main()
