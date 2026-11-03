import logging
import re

import pytest

from pii_toolbox.core import Detector, Document, EntityGroup, IdentityResolver, PiiType, Placeholder, Span
from pii_toolbox.session import AngleBracketFormat, InMemoryPlaceholderStore, Session


class FakeDetector(Detector):
    """Finds each listed phrase wherever it occurs, labelled by a per-phrase label."""

    def __init__(self, phrases: dict[str, tuple[PiiType, str]]) -> None:
        self._phrases = phrases

    @property
    def name(self) -> str:
        return "fake"

    @property
    def languages(self):
        return None

    @property
    def detects(self):
        return frozenset(pii_type for pii_type, _ in self._phrases.values())

    def load_models(self) -> None:
        pass

    def detect(self, document: Document) -> list[Span]:
        spans = []
        for phrase, (pii_type, label) in self._phrases.items():
            for match in re.finditer(re.escape(phrase), document.text):
                spans.append(Span(match.start(), match.end(), pii_type, label, ("fake",), None))
        return sorted(spans, key=lambda s: s.start)


class FakeResolver(IdentityResolver):
    """Puts every detected PERSON span into one group whose canonical span is the longest."""

    @property
    def name(self) -> str:
        return "fake-resolver"

    def resolve(self, document, spans):
        people = [s for s in spans if s.pii_type is PiiType.PERSON]
        if not people:
            return []
        return [EntityGroup(tuple(people), max(people, key=lambda s: s.end - s.start))]


PHRASES = {
    "David Parker": (PiiType.PERSON, "PERSON"),
    "Dave": (PiiType.PERSON, "PERSON"),
    "Maria Stone": (PiiType.PERSON, "PERSON"),
    "dave@example.test": (PiiType.IDENTIFIER, "EMAIL_ADDRESS"),
}


def make_session(resolver=None, store=None) -> Session:
    return Session(
        FakeDetector(PHRASES), store or InMemoryPlaceholderStore(), AngleBracketFormat(), "en", resolver, "case-1"
    )


def test_texts_in_one_session_share_numbering():
    session = make_session()
    assert session.mask("David Parker met us.").masked_text == "<PERSON_1> met us."
    assert session.mask("Maria Stone met us.").masked_text == "<PERSON_2> met us."


def test_same_original_gets_same_placeholder_across_texts():
    session = make_session()
    session.mask("David Parker met us.")
    session.mask("Maria Stone left.")
    assert session.mask("Later David Parker left.").masked_text == "Later <PERSON_1> left."


def test_new_session_restarts_numbering():
    first = make_session()
    first.mask("Maria Stone left.")
    first.mask("David Parker left.")
    assert make_session().mask("David Parker left.").masked_text == "<PERSON_1> left."


def test_resolver_groups_name_forms_under_one_placeholder():
    session = make_session(FakeResolver())
    result = session.mask("David Parker joined. Dave was glad.")
    assert result.masked_text == "<PERSON_1> joined. <PERSON_1> was glad."
    assert session.store.original("case-1", Placeholder("PERSON", 1)) == "David Parker"


def test_mask_without_resolver_keeps_forms_apart():
    result = make_session().mask("David Parker joined. Dave was glad.")
    assert result.masked_text == "<PERSON_1> joined. <PERSON_2> was glad."


def test_detailed_label_is_used_and_adjacent_spans_replace_from_the_end():
    result = make_session().mask("Dave@x dave@example.test David Parker")
    assert result.masked_text == "<PERSON_1>@x <EMAIL_ADDRESS_1> <PERSON_2>"


def test_minted_lists_only_this_calls_placeholders():
    session = make_session()
    assert session.mask("David Parker").minted == (Placeholder("PERSON", 1),)
    second = session.mask("David Parker and Maria Stone and dave@example.test")
    assert second.minted == (Placeholder("PERSON", 2), Placeholder("EMAIL_ADDRESS", 1))


def test_no_original_in_logs_stdout_or_files(caplog, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)
    session = make_session(FakeResolver())
    session.mask("David Parker wrote to dave@example.test. Dave replied.")
    originals = ["David Parker", "Dave", "dave@example.test"]
    captured = capsys.readouterr()
    for original in originals:
        assert original not in caplog.text
        assert original not in captured.out + captured.err
    assert list(tmp_path.iterdir()) == []
    assert "<PERSON_1>" in caplog.text
    assert "<EMAIL_ADDRESS_1>" in caplog.text
    assert "3 spans" in caplog.text


def test_close_discards_the_mapping():
    session = make_session()
    placeholder = session.mask("David Parker").minted[0]
    session.close()
    assert session.store.original("case-1", placeholder) is None
    assert session.store.labels("case-1") == frozenset()


def test_exception_inside_with_block_discards_the_mapping():
    session = make_session()
    placeholder = None
    with pytest.raises(RuntimeError):
        with session:
            placeholder = session.mask("David Parker").minted[0]
            raise RuntimeError("boom")
    assert session.store.original("case-1", placeholder) is None
    assert session.store.labels("case-1") == frozenset()


def test_result_reports_the_placeholder_each_span_received():
    result = make_session(FakeResolver()).mask("David Parker joined. Dave was glad.")
    assert {(span.start, span.end): placeholder for span, placeholder in result.placeholder_by_span.items()} == {
        (0, 12): Placeholder("PERSON", 1),
        (21, 25): Placeholder("PERSON", 1),
    }
