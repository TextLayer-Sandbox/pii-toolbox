from pii_toolbox.core import Detector, Document, PiiType, Span
from pii_toolbox.session import AngleBracketFormat, InMemoryPlaceholderStore, Session


class NoDetector(Detector):
    @property
    def name(self) -> str:
        return "none"

    @property
    def languages(self):
        return None

    @property
    def detects(self):
        return frozenset()

    def load_models(self) -> None:
        pass

    def detect(self, document: Document) -> list[Span]:
        return []


def make_session(session_id: str = "case-1") -> Session:
    return Session(NoDetector(), InMemoryPlaceholderStore(), AngleBracketFormat(), "en", None, session_id)


def session_with_people(count: int) -> Session:
    session = make_session()
    for index in range(1, count + 1):
        session.store.get_or_mint(session.id, "PERSON", f"Person Number{index}")
    return session


def test_restore_leaves_inexact_forms_untouched():
    session = session_with_people(1)
    for token in ["PERSON_1", "person_1", "Person_1", "<person_1>", "<PERSON_01>"]:
        result = session.restore(token)
        assert result.restored_text == token
        assert result.leftovers == ()


def test_restore_matches_whole_tokens_only():
    session = session_with_people(10)
    assert session.restore("<PERSON_10>").restored_text == "Person Number10"
    assert session.restore("<PERSON_1> and <PERSON_10>").restored_text == "Person Number1 and Person Number10"


def test_token_is_restored_whole_next_to_a_letter_or_digit():
    session = session_with_people(1)
    assert session.restore("x<PERSON_1>").restored_text == "xPerson Number1"
    assert session.restore("5<PERSON_1>").restored_text == "5Person Number1"
    assert session.restore("<PERSON_1>s lawyer").restored_text == "Person Number1s lawyer"


def test_look_alike_passes_through_and_is_not_a_leftover():
    session = session_with_people(1)
    result = session.restore("ask TEAM_7 about <PERSON_1>")
    assert result.restored_text == "ask TEAM_7 about Person Number1"
    assert result.leftovers == ()


def test_unminted_number_of_minted_label_is_a_leftover():
    session = session_with_people(1)
    result = session.restore("<PERSON_9> and <PERSON_1>")
    assert result.restored_text == "<PERSON_9> and Person Number1"
    assert result.leftovers == ("<PERSON_9>",)


def test_leftovers_keep_text_order():
    session = session_with_people(1)
    assert session.restore("<PERSON_8> then <PERSON_9>").leftovers == ("<PERSON_8>", "<PERSON_9>")


def test_other_sessions_placeholders_are_not_restored():
    first = session_with_people(1)
    second = make_session("case-2")
    assert second.restore("<PERSON_1>").restored_text == "<PERSON_1>"
    assert second.restore("<PERSON_1>").leftovers == ()
    assert first.restore("<PERSON_1>").restored_text == "Person Number1"


def test_empty_text_restores_to_empty_text():
    result = session_with_people(1).restore("")
    assert result.restored_text == ""
    assert result.leftovers == ()


def test_span_with_free_text_detector_label_masks_by_its_type_and_restores():
    case_number_span = Span(5, 11, PiiType.IDENTIFIER, "case number", ("test",), None)

    class CaseNumberDetector(NoDetector):
        def detect(self, document: Document) -> list[Span]:
            return [case_number_span]

    session = Session(CaseNumberDetector(), InMemoryPlaceholderStore(), AngleBracketFormat(), "en", None, "case-1")
    masked = session.mask("Case 12-345 open").masked_text
    assert masked == "Case <IDENTIFIER_1> open"
    assert session.restore(masked).restored_text == "Case 12-345 open"
