"""Test of the real Maverick worker; needs the downloaded model and no network."""

import pytest

from pii_toolbox.coref.resolver import MaverickResolver
from pii_toolbox.core import Document, PiiType, Span
from pii_toolbox.core.model_dir import default_model_dir

# Resolved at import so the per-test sandbox home does not hide the downloaded models.
MODEL_DIR = default_model_dir()

pytestmark = pytest.mark.model

PASSAGE = "David Parker joined in May. Dave was glad, and Mr. Parker thanked the team."


def test_name_forms_of_one_person_share_a_group():
    document = Document(id="d1", text=PASSAGE, language="en")
    dave_start = PASSAGE.index("Dave")
    detected = [
        Span(0, 12, PiiType.PERSON, "PERSON", ("fake",), 0.9),
        Span(dave_start, dave_start + 4, PiiType.PERSON, "PERSON", ("fake",), 0.9),
    ]
    with MaverickResolver(model_dir=MODEL_DIR) as resolver:
        groups = resolver.resolve(document, detected)
    (group,) = groups
    assert [PASSAGE[span.start : span.end] for span in group.spans] == ["David Parker", "Dave", "Mr. Parker"]
