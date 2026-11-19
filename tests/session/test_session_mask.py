import logging
import re

import pytest

from pii_toolbox.core import Detector, Document, PersonMentions, IdentityResolver, PiiType, Placeholder, Span
from pii_toolbox.session import AngleBracketFormat, InMemoryPlaceholderStore, Session


class FakeDetector(Detector):
    """Finds each listed phrase wherever it occurs, labelled by a per-phrase label."""

    def __init__(self, phrases: dict[str, tuple[PiiType, str]]) -> None:
        self._phrases = phrases

    @property
    def name(self) -> str:
        return "fake"

    @property
    def languages(self) -> frozenset[str] | None:
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
        return [PersonMentions(tuple(people), max(people, key=lambda s: s.end - s.start))]


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


def test_group_reuses_placeholder_a_member_already_holds():
    session = make_session(FakeResolver())
    assert session.mask("Dave left.").masked_text == "<PERSON_1> left."
    second = session.mask("David Parker joined. Dave was glad.")
    assert second.masked_text == "<PERSON_1> joined. <PERSON_1> was glad."
    assert second.minted == ()


def test_restore_after_group_joins_earlier_placeholder_returns_fullest_form_everywhere():
    session = make_session(FakeResolver())
    session.mask("Dave left.")
    masked = session.mask("David Parker joined. Dave was glad.").masked_text
    assert session.restore(masked).restored_text == "David Parker joined. David Parker was glad."


def test_mask_without_resolver_keeps_forms_apart():
    result = make_session().mask("David Parker joined. Dave was glad.")
    assert result.masked_text == "<PERSON_1> joined. <PERSON_2> was glad."


def test_placeholder_uses_the_common_type_and_adjacent_spans_replace_from_the_end():
    result = make_session().mask("Dave@x dave@example.test David Parker")
    assert result.masked_text == "<PERSON_1>@x <IDENTIFIER_1> <PERSON_2>"


def test_minted_lists_only_this_calls_placeholders():
    session = make_session()
    assert session.mask("David Parker").minted == (Placeholder("PERSON", 1),)
    second = session.mask("David Parker and Maria Stone and dave@example.test")
    assert second.minted == (Placeholder("PERSON", 2), Placeholder("IDENTIFIER", 1))


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
    assert "<IDENTIFIER_1>" in caplog.text
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
    assert placeholder is not None
    assert session.store.original("case-1", placeholder) is None
    assert session.store.labels("case-1") == frozenset()


def test_result_reports_the_placeholder_each_span_received():
    result = make_session(FakeResolver()).mask("David Parker joined. Dave was glad.")
    assert {(span.start, span.end): placeholder for span, placeholder in result.placeholder_by_span.items()} == {
        (0, 12): Placeholder("PERSON", 1),
        (21, 25): Placeholder("PERSON", 1),
    }


class FixedSpanDetector(Detector):
    """Returns the given spans for any text."""

    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans

    @property
    def name(self) -> str:
        return "fixed"

    @property
    def languages(self):
        return None

    @property
    def detects(self):
        return frozenset(span.pii_type for span in self._spans)

    def load_models(self) -> None:
        pass

    def detect(self, document: Document) -> list[Span]:
        return list(self._spans)


class AddingResolver(IdentityResolver):
    """Groups the first span with an extra name-form span at fixed offsets."""

    def __init__(self, added: Span) -> None:
        self._added = added

    @property
    def name(self) -> str:
        return "adding-resolver"

    def resolve(self, document, spans):
        return [PersonMentions((spans[0], self._added), spans[0])]


def test_overlapping_detected_spans_are_merged_before_substitution():
    spans = [
        Span(0, 5, PiiType.PERSON, "PERSON", ("fixed",), 0.9),
        Span(1, 2, PiiType.PERSON, "PERSON", ("fixed",), 0.5),
    ]
    session = Session(FixedSpanDetector(spans), InMemoryPlaceholderStore(), AngleBracketFormat(), "en", None, "case-1")
    result = session.mask("Alice x")
    assert result.masked_text == "<PERSON_1> x"
    assert "Alice" not in result.masked_text


def test_resolver_added_span_overlapping_other_type_is_merged():
    person = Span(0, 5, PiiType.PERSON, "PERSON", ("fixed",), 0.9)
    organization = Span(11, 20, PiiType.ORGANIZATION, "ORGANIZATION", ("fixed",), 0.9)
    added = Span(16, 24, PiiType.PERSON, "PERSON", ("adding-resolver",), None)
    session = Session(
        FixedSpanDetector([person, organization]),
        InMemoryPlaceholderStore(),
        AngleBracketFormat(),
        "en",
        AddingResolver(added),
        "case-1",
    )
    text = "Alice sees Acme Corp Bob ok"
    result = session.mask(text)
    assert "Acme" not in result.masked_text
    assert "Bob" not in result.masked_text
    assert result.masked_text == "<PERSON_1> sees <ORGANIZATION_1> ok"


def test_placeholder_look_alike_in_input_round_trips_exactly():
    session = make_session()
    text = "The log shows <PERSON_1> failed. Email David Parker."
    result = session.mask(text)
    assert result.masked_text == "The log shows <IDENTIFIER_1> failed. Email <PERSON_1>."
    assert session.restore(result.masked_text).restored_text == text


def test_placeholder_look_alike_typed_after_its_number_was_minted_round_trips_exactly():
    session = make_session()
    first = session.mask("Email David Parker.")
    assert first.masked_text == "Email <PERSON_1>."
    text = "Ignore <PERSON_1> here; Maria Stone replied."
    second = session.mask(text)
    assert second.masked_text == "Ignore <IDENTIFIER_1> here; <PERSON_2> replied."
    assert session.restore(second.masked_text).restored_text == text
    assert session.restore(first.masked_text).restored_text == "Email David Parker."


def test_text_not_in_the_rendered_placeholder_form_round_trips_unchanged():
    session = make_session()
    session.mask("Email David Parker.")
    text = "Ignore person_1, Person_1, PERSON_1, <person_1>, <PERSON_01>, user_id_2 and item_3 here."
    result = session.mask(text)
    assert result.masked_text == text
    assert session.restore(result.masked_text).restored_text == text


class SurnameDetector(FakeDetector):
    """Finds full names and the bare surname without letting the surname match inside a full name."""

    def __init__(self) -> None:
        super().__init__({"John Smith": (PiiType.PERSON, "PERSON")})

    def detect(self, document: Document) -> list[Span]:
        return [
            Span(match.start(), match.end(), PiiType.PERSON, "PERSON", ("fake",), None)
            for match in re.finditer(r"John Smith|Anna Smith|Smith", document.text)
        ]


def make_surname_session() -> Session:
    return Session(SurnameDetector(), InMemoryPlaceholderStore(), AngleBracketFormat(), "en", FakeResolver(), "case-1")


def test_new_person_sharing_a_short_form_gets_own_placeholder_for_the_whole_group():
    session = make_surname_session()
    first = session.mask("John Smith met Smith.")
    second = session.mask("Anna Smith met Smith.")
    assert first.masked_text == "<PERSON_1> met <PERSON_1>."
    assert second.masked_text == "<PERSON_2> met <PERSON_2>."
    assert session.restore(second.masked_text).restored_text == "Anna Smith met Anna Smith."
    assert session.restore(first.masked_text).restored_text == "John Smith met John Smith."


def test_short_form_stays_with_its_first_holder_after_another_person_shares_it():
    session = make_surname_session()
    session.mask("John Smith met Smith.")
    session.mask("Anna Smith met Smith.")
    assert session.mask("Smith left.").masked_text == "<PERSON_1> left."


def test_same_full_name_in_later_text_keeps_one_placeholder_with_shared_short_form():
    session = make_surname_session()
    session.mask("John Smith met Smith.")
    assert session.mask("John Smith called Smith.").masked_text == "<PERSON_1> called <PERSON_1>."


class LanguageLimitedDetector(FakeDetector):
    """A fake detector that supports only the given languages."""

    def __init__(self, languages: frozenset[str]) -> None:
        super().__init__({})
        self._languages = languages

    @property
    def languages(self) -> frozenset[str] | None:
        return self._languages


def test_session_refuses_detector_without_its_language():
    detector = LanguageLimitedDetector(frozenset({"en"}))
    with pytest.raises(ValueError, match="does not support language fr"):
        Session(detector, InMemoryPlaceholderStore(), AngleBracketFormat(), "fr")


def test_session_accepts_detector_supporting_its_language():
    detector = LanguageLimitedDetector(frozenset({"en", "fr"}))
    session = Session(detector, InMemoryPlaceholderStore(), AngleBracketFormat(), "fr")
    assert session.language == "fr"


def test_session_accepts_detector_supporting_every_language():
    session = Session(FakeDetector(PHRASES), InMemoryPlaceholderStore(), AngleBracketFormat(), "fr")
    assert session.language == "fr"
