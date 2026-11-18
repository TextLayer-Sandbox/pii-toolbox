"""Tests for entity-group building from fabricated mention clusters."""

import pytest

from pii_toolbox.coref import build_groups
from pii_toolbox.core import Document, PiiType, Span

TEXT = "David Parker met Anna. Dave said he was late. Mr. Parker left. Anna smiled at her."
DOC = Document(id="d1", text=TEXT, language="en")


def pos(text: str, needle: str, occurrence: int = 0) -> tuple[int, int]:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(needle, start + 1)
    return start, start + len(needle)


def person(start: int, end: int, source: str = "presidio") -> Span:
    return Span(start, end, PiiType.PERSON, "PERSON", (source,), 0.9)


DAVID = pos(TEXT, "David Parker")
DAVE = pos(TEXT, "Dave")
HE = pos(TEXT, "he")
MR_PARKER = pos(TEXT, "Mr. Parker")
ANNA = pos(TEXT, "Anna")
HER = pos(TEXT, "her")


def test_cluster_without_detected_person_is_discarded():
    assert build_groups(DOC, [], [[DAVID, DAVE]]) == []
    org = Span(*DAVID, PiiType.ORGANIZATION, "ORG", ("x",), 0.5)
    assert build_groups(DOC, [org], [[DAVID, DAVE]]) == []


def test_missed_name_forms_are_added_with_resolver_source():
    detected = person(*DAVID)
    (group,) = build_groups(DOC, [detected], [[DAVID, DAVE, MR_PARKER]])
    assert [(s.start, s.end) for s in group.spans] == [DAVID, DAVE, MR_PARKER]
    assert group.spans[0] == detected
    for added in group.spans[1:]:
        assert added.source == ("maverick",)
        assert added.score is None
        assert added.pii_type is PiiType.PERSON
    assert group.canonical == detected


def test_pronouns_and_non_name_mentions_are_dropped():
    (group,) = build_groups(DOC, [person(*DAVID)], [[DAVID, DAVE, HE]])
    assert [(s.start, s.end) for s in group.spans] == [DAVID, DAVE]


def test_cluster_of_pronouns_around_one_detected_name_keeps_only_the_name():
    (group,) = build_groups(DOC, [person(*ANNA)], [[ANNA, HER]])
    assert [(s.start, s.end) for s in group.spans] == [ANNA]
    assert group.canonical == group.spans[0]


def test_canonical_is_longest_name_form_even_when_added():
    text = "Dave met Dr. Jane Smith. Jane left."
    doc = Document(id="d2", text=text, language="en")
    short = pos(text, "Dave")
    full = pos(text, "Dr. Jane Smith")
    (group,) = build_groups(doc, [person(*short)], [[short, full]])
    assert (group.canonical.start, group.canonical.end) == full
    assert group.canonical.source == ("maverick",)


def test_groups_come_out_ordered_and_separate():
    groups = build_groups(DOC, [person(*DAVID), person(*ANNA)], [[ANNA, HER], [DAVID, DAVE]])
    assert [g.canonical.start for g in groups] == [DAVID[0], ANNA[0]]


def test_title_without_surname_is_not_a_name_form():
    text = "Anna met Mr. then Dr. left."
    doc = Document(id="d3", text=text, language="en")
    anna = pos(text, "Anna")
    (group,) = build_groups(doc, [person(*anna)], [[anna, pos(text, "Mr."), pos(text, "Dr.")]])
    assert len(group.spans) == 1


def test_mention_outside_document_raises():
    with pytest.raises(ValueError):
        build_groups(DOC, [person(*DAVID)], [[DAVID, (0, len(TEXT) + 1)]])


def test_mention_overlapping_two_detected_names_links_neither():
    text = "John Smith's wife Mary Jones said she left."
    doc = Document(id="d2", text=text, language="en")
    john = pos(text, "John Smith")
    mary = pos(text, "Mary Jones")
    both = pos(text, "John Smith's wife Mary Jones")
    groups = build_groups(doc, [person(*john), person(*mary)], [[both, pos(text, "she")], [john], [mary]])
    assert [[(s.start, s.end) for s in g.spans] for g in groups] == [[john], [mary]]


def test_mention_inside_detected_name_still_links():
    text = "John Smith arrived. Smith sat down."
    doc = Document(id="d3", text=text, language="en")
    john = pos(text, "John Smith")
    smith = pos(text, "Smith", 1)
    (group,) = build_groups(doc, [person(*john)], [[john, smith]])
    assert [(s.start, s.end) for s in group.spans] == [john, smith]
