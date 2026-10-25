"""Tests for the always-recognise names detector."""

import pytest

from pii_toolbox.core import Document, PiiType
from pii_toolbox.detection.known_names import KnownNamesDetector
from pii_toolbox.detection.registry import FixedDetectorRegistry, build_from_config


def detect(detector, text):
    spans = detector.detect(Document(id="d", text=text, language="en"))
    return [text[span.start : span.end] for span in spans]


def test_listed_name_is_found_as_a_person_span():
    text = "Yesterday Zephyrine Quill signed the form."
    spans = KnownNamesDetector(["Zephyrine Quill"]).detect(Document(id="d", text=text, language="en"))
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "Zephyrine Quill"
    assert spans[0].pii_type == PiiType.PERSON
    assert spans[0].source == ("known_names",)


def test_every_occurrence_is_found():
    assert detect(KnownNamesDetector(["Rose"]), "Rose met Rose. Then Rose left.") == ["Rose"] * 3


def test_capitalised_only_is_the_default_and_skips_lowercase():
    assert detect(KnownNamesDetector(["Rose"]), "A rose and Rose.") == ["Rose"]


def test_case_insensitive_matches_lowercase():
    assert detect(KnownNamesDetector(["Rose"], case_insensitive=True), "A rose and Rose.") == ["rose", "Rose"]


@pytest.mark.parametrize("case_insensitive", [False, True])
@pytest.mark.parametrize("text", ["Rosa came.", "Roze came.", "The Rosewood table.", "Primrose came.", "Rosé came."])
def test_typos_and_longer_words_are_never_matched(text, case_insensitive):
    assert detect(KnownNamesDetector(["Rose"], case_insensitive), text) == []


def test_name_next_to_punctuation_matches():
    assert detect(KnownNamesDetector(["Rose"]), '"Rose," she said (Rose).') == ["Rose", "Rose"]


def test_longer_listed_name_wins_over_its_prefix():
    assert detect(KnownNamesDetector(["Rose", "Rose Tyler"]), "Rose Tyler and Rose") == ["Rose Tyler", "Rose"]


def test_empty_list_finds_nothing():
    assert detect(KnownNamesDetector([]), "Rose") == []


def test_names_file_has_one_name_per_line(tmp_path):
    names_file = tmp_path / "names.txt"
    names_file.write_text("Rose\n\n  Zephyrine Quill  \n", encoding="utf-8")
    detector = KnownNamesDetector.from_file(names_file)
    assert detect(detector, "Rose and Zephyrine Quill") == ["Rose", "Zephyrine Quill"]


def test_registered_under_known_names_and_configured_through_the_registry():
    registry = FixedDetectorRegistry(known_names=["Rose"], known_names_case_insensitive=True)
    detector = registry.build("known_names")
    assert detector.name == "known_names"
    assert detect(detector, "rose") == ["rose"]
    assert [d.name for d in build_from_config(["known_names"], registry)] == ["known_names"]
