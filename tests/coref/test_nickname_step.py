"""Tests for the nickname step, using hand-built groups so no co-reference model is needed."""

from pathlib import Path

import pytest

from pii_toolbox.coref.nickname_step import NicknameConfig, apply_nickname_step
from pii_toolbox.core import Document, EntityGroup, PiiType, Span

DEFAULT = NicknameConfig()


def locate(text: str, needle: str, occurrence: int = 0) -> Span:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(needle, start + 1)
    return Span(start, start + len(needle), PiiType.PERSON, "PERSON", ("presidio",), 0.9)


def group_of(*spans: Span) -> EntityGroup:
    return EntityGroup(spans=spans, canonical=max(spans, key=lambda span: span.end - span.start))


def positions(group: EntityGroup) -> list[tuple[int, int]]:
    return [(span.start, span.end) for span in group.spans]


def test_dave_joins_the_david_parker_group():
    text = "David Parker met Anna. Dave said hello. Mr. Parker left."
    document = Document(id="d1", text=text, language="en")
    david, dave, mr_parker = locate(text, "David Parker"), locate(text, "Dave"), locate(text, "Mr. Parker")
    groups = [group_of(david, mr_parker)]
    result = apply_nickname_step(document, [david, dave, mr_parker], groups, DEFAULT)
    assert len(result) == 1
    assert positions(result[0]) == [(david.start, david.end), (dave.start, dave.end), (mr_parker.start, mr_parker.end)]
    assert result[0].canonical == david


def test_nicknames_do_not_link_to_each_other():
    text = "Matt arrived early. Matty stayed late."
    document = Document(id="d2", text=text, language="en")
    matt, matty = locate(text, "Matt"), locate(text, "Matty")
    assert apply_nickname_step(document, [matt, matty], [], DEFAULT) == []


def test_matt_and_matty_join_a_matthew_group_through_matthew_only():
    text = "Matthew Hale wrote. Matt agreed. Matty laughed."
    document = Document(id="d3", text=text, language="en")
    matthew, matty = locate(text, "Matthew Hale"), locate(text, "Matty")
    matt = Span(text.index("Matt agreed"), text.index("Matt agreed") + 4, PiiType.PERSON, "PERSON", ("presidio",), 0.9)
    result = apply_nickname_step(document, [matthew, matt, matty], [group_of(matthew)], DEFAULT)
    assert len(result) == 1
    assert result[0].canonical == matthew
    assert len(result[0].spans) == 3


def test_a_name_already_in_a_group_stays_there():
    text = "David Parker met Dave Smith."
    document = Document(id="d4", text=text, language="en")
    parker, smith = locate(text, "David Parker"), locate(text, "Dave Smith")
    groups = [group_of(parker), group_of(smith)]
    result = apply_nickname_step(document, [parker, smith], groups, DEFAULT)
    assert [positions(group) for group in result] == [positions(group) for group in groups]


def test_a_name_with_a_different_surname_does_not_join():
    text = "David Parker met Dave Smith."
    document = Document(id="d5", text=text, language="en")
    parker, smith = locate(text, "David Parker"), locate(text, "Dave Smith")
    result = apply_nickname_step(document, [parker, smith], [group_of(parker)], DEFAULT)
    assert positions(result[0]) == [(parker.start, parker.end)]


def test_a_name_matching_two_groups_stays_unlinked():
    text = "David Parker met David Lee. Dave left."
    document = Document(id="d6", text=text, language="en")
    parker, lee, dave = locate(text, "David Parker"), locate(text, "David Lee"), locate(text, "Dave")
    result = apply_nickname_step(document, [parker, lee, dave], [group_of(parker), group_of(lee)], DEFAULT)
    assert all((dave.start, dave.end) not in positions(group) for group in result)


def test_a_configured_list_replaces_the_default(tmp_path: Path):
    text = "David Parker met Dave. Davo laughed."
    document = Document(id="d7", text=text, language="en")
    david, dave, davo = locate(text, "David Parker"), locate(text, "Dave"), locate(text, "Davo")
    replacement = tmp_path / "custom.csv"
    replacement.write_text("davo,david\n", encoding="utf-8")
    result = apply_nickname_step(document, [david, dave, davo], [group_of(david)], NicknameConfig(replacement))
    assert positions(result[0]) == [(david.start, david.end), (davo.start, davo.end)]


def test_a_configured_list_that_breaks_the_shape_is_rejected(tmp_path: Path):
    broken = tmp_path / "broken.csv"
    broken.write_text("davo,david\ndavo,dave\n", encoding="utf-8")
    document = Document(id="d8", text="David met Davo.", language="en")
    with pytest.raises(ValueError):
        apply_nickname_step(document, [], [], NicknameConfig(broken))


def test_with_the_list_off_dave_stays_unlinked():
    text = "David Parker met Dave."
    document = Document(id="d9", text=text, language="en")
    david, dave = locate(text, "David Parker"), locate(text, "Dave")
    result = apply_nickname_step(document, [david, dave], [group_of(david)], NicknameConfig(None))
    assert positions(result[0]) == [(david.start, david.end)]
