import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_round_trip_example_prints_masked_reply_and_restored_lines():
    completed = subprocess.run(
        [sys.executable, "examples/round_trip.py"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines() == [
        "masked: <PERSON_1> wrote to <EMAIL_ADDRESS_1>. <PERSON_1> replied.",
        "reply: Summary: <PERSON_1> emailed <EMAIL_ADDRESS_1>. <PERSON_1> replied.",
        "restored: Summary: David Parker emailed dave@acme.example. David Parker replied.",
    ]
