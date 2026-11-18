"""Runnable multi-turn example: one session masks a whole conversation and restores a later reply.

Run from the repository root with `uv run python examples/multi_turn_session.py`.
"""

from pii_toolbox.detection.known_names import KnownNamesDetector
from pii_toolbox.session import AngleBracketFormat, InMemoryPlaceholderStore, Session

PEOPLE = ("Ana Example", "Ben Sample")

USER_TURNS = (
    "Ana Example is leading the audit.",
    "Ben Sample reports to Ana Example.",
    "Who should Ben Sample ask about the budget?",
)

# A model sees only placeholders, yet can refer back to a person introduced in an earlier turn.
MODEL_REPLY = "<PERSON_2> should ask <PERSON_1>, who is leading the audit."


def main() -> None:
    detector = KnownNamesDetector(PEOPLE)
    with Session(detector, InMemoryPlaceholderStore(), AngleBracketFormat(), "en") as session:
        for number, turn in enumerate(USER_TURNS, start=1):
            print(f"turn {number}: {session.mask(turn).masked_text}")
        restored = session.restore(MODEL_REPLY)

    print(f"reply: {MODEL_REPLY}")
    print(f"restored: {restored.restored_text}")


if __name__ == "__main__":
    main()
