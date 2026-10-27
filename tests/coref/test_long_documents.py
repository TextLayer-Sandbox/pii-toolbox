"""Tests that the co-reference worker handles documents longer than one encoder pass can afford."""

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pii_toolbox.coref.resolver import DEFAULT_REPLY_TIMEOUT_SECONDS, MaverickResolver, WORKER_DIR
from pii_toolbox.core import Document, PiiType, Span
from pii_toolbox.evaluation.model_manifest import default_model_dir

# Resolved at import so the per-test sandbox home does not hide the downloaded models.
MODEL_DIR = default_model_dir()

LONG_DOCUMENT_WORDS = 10_000


def load_worker_module():
    spec = importlib.util.spec_from_file_location("coref_worker_server", WORKER_DIR / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMaverick:
    """Stands in for the model object: records how the encoder network is called and reports drifting offsets."""

    def __init__(self, subtokens: int) -> None:
        self.subtokens = subtokens
        self.forward_calls: list[dict] = []
        self.model = SimpleNamespace(forward=self.forward)

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)

    def preprocess(self, sample, speakers=None):
        tokens = sample.split()
        drifting_offsets = [(index, index) for index in range(len(tokens))]
        return tokens, [len(tokens)], ["-"] * len(tokens), drifting_offsets

    def predict(self, sample):
        tokens, _, _, offsets = self.preprocess(sample)
        self.model.forward(stage="test", input_ids=np.zeros((1, self.subtokens)))
        return {"clusters_char_offsets": [[offsets[0], offsets[-1]]]}


def test_positions_stay_exact_across_blank_lines():
    worker = load_worker_module()
    text = "Zorvan Thessaly spoke.\n\n\nThe clerk left.\n\nMr. Thessaly replied."
    predict = worker.maverick_predict_function(FakeMaverick(10))
    ((first, last),) = predict(text)
    assert text[first[0] : first[1]] == "Zorvan"
    assert text[last[0] : last[1]] == "replied."


def long_document(words: int) -> str:
    subjects = ["The committee", "The auditor", "A neighbour", "The clerk", "The delegation", "The engineer"]
    verbs = ["reviewed", "postponed", "described", "measured", "approved", "questioned"]
    objects = ["the harbour plan", "the quarterly ledger", "the bridge report", "the water tariff", "the market permit"]
    lines = ["Zorvan Thessaly opened the meeting in Brindlemoor and read the agenda aloud."]
    count = len(lines[0].split())
    index = 0
    while count < words - 20:
        sentence = f"{subjects[index % 6]} {verbs[(index // 6) % 6]} {objects[(index // 36) % 5]} after a long discussion."
        lines.append(sentence)
        if index % 12 == 11:
            lines.append("")
        count += len(sentence.split())
        index += 1
    lines.append("At the close, Mr. Thessaly thanked everyone, and Zorvan said he would sign the minutes himself.")
    return "\n".join(lines)


@pytest.mark.model
def test_real_worker_resolves_a_ten_thousand_word_document_in_time():
    text = long_document(LONG_DOCUMENT_WORDS)
    document = Document(id="long", text=text, language="en")
    mister = text.index("Mr. Thessaly") + len("Mr. ")
    zorvan = text.rindex("Zorvan")
    spans = [
        Span(0, len("Zorvan Thessaly"), PiiType.PERSON, "PERSON", ("fake",), 0.9),
        Span(mister, mister + len("Thessaly"), PiiType.PERSON, "PERSON", ("fake",), 0.9),
        Span(zorvan, zorvan + len("Zorvan"), PiiType.PERSON, "PERSON", ("fake",), 0.9),
    ]
    with MaverickResolver(model_dir=MODEL_DIR) as resolver:
        started = time.monotonic()
        groups = resolver.resolve(document, spans)
        elapsed = time.monotonic() - started
    assert elapsed < DEFAULT_REPLY_TIMEOUT_SECONDS
    assert len(text.split()) >= LONG_DOCUMENT_WORDS - 20
    later_group = next(group for group in groups if any(span.start == zorvan for span in group.spans))
    assert {text[span.start : span.end] for span in later_group.spans} >= {"Thessaly", "Zorvan"}
