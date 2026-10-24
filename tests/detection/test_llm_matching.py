import pytest

from pii_toolbox.core import PiiType
from pii_toolbox.detection.llm_matching import match_items

TEXT = 'Dr. Ana Souza said "call me on 555-0100" today'


def test_plain_item_maps_to_offsets():
    spans, unmatched = match_items(TEXT, [("Ana Souza", "PERSON")])
    assert unmatched == []
    assert len(spans) == 1
    assert TEXT[spans[0].start : spans[0].end] == "Ana Souza"
    assert spans[0].pii_type is PiiType.PERSON
    assert spans[0].detector_label == "PERSON"


@pytest.mark.parametrize(
    ("text", "item", "expected"),
    [
        (TEXT, "ana souza", "Ana Souza"),
        (TEXT, "ANA SOUZA", "Ana Souza"),
        (TEXT, "Ana  Souza", "Ana Souza"),
        (TEXT, "Ana\nSouza", "Ana Souza"),
        ("Dr. Ana  Souza said hi", "Ana Souza", "Ana  Souza"),
        ("Dr. Ana\nSouza said hi", "Ana Souza", "Ana\nSouza"),
        (TEXT, "“call me on 555-0100”", '"call me on 555-0100"'),
        ("He said “hello there”", '"hello there"', "“hello there”"),
        ("It’s Ana Souza’s", "Ana Souza's", "Ana Souza’s"),
    ],
)
def test_variants_map_to_original_text(text, item, expected):
    spans, unmatched = match_items(text, [(item, "PERSON")])
    assert unmatched == []
    assert text[spans[0].start : spans[0].end] == expected


def test_repeated_value_maps_to_each_occurrence_in_order():
    text = "call 555-0100 or 555-0100 now"
    spans, unmatched = match_items(text, [("555-0100", "IDENTIFIER"), ("555-0100", "IDENTIFIER")])
    assert unmatched == []
    assert [(s.start, s.end) for s in spans] == [(5, 13), (17, 25)]


def test_repeated_item_beyond_occurrences_is_unmatched():
    spans, unmatched = match_items("one 555-0100 only", [("555-0100", "IDENTIFIER")] * 2)
    assert len(spans) == 1
    assert unmatched == [("555-0100", "IDENTIFIER")]


def test_absent_item_is_unmatched_and_produces_no_span():
    spans, unmatched = match_items(TEXT, [("Maria Lopez", "PERSON")])
    assert spans == []
    assert unmatched == [("Maria Lopez", "PERSON")]


def test_unknown_label_is_unmatched():
    spans, unmatched = match_items(TEXT, [("Ana Souza", "SPECIES")])
    assert spans == []
    assert unmatched == [("Ana Souza", "SPECIES")]
