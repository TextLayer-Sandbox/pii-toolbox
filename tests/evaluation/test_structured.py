from types import SimpleNamespace

from pii_toolbox.datasets.structured_pii import LabelledSpan, StructuredDocument
from pii_toolbox.evaluation.structured import (
    Finding,
    build_recognisers,
    build_result,
    count_documents,
    find_entities,
    item_outcome,
    merge_counts,
    summarise,
)


class StubRecogniser:
    supported_entities = ["CREDIT_CARD", "DATE_TIME"]

    def __init__(self, results):
        self._results = results

    def analyze(self, text, entities):
        return [SimpleNamespace(entity_type=entity, start=start, end=end) for entity, start, end in self._results]


def test_overlap_by_one_character_counts_as_found():
    assert item_outcome("credit_debit_card", 10, 20, [Finding("DATE_TIME", 19, 30)]) == (True, False)


def test_adjacent_ranges_do_not_overlap():
    assert item_outcome("credit_debit_card", 10, 20, [Finding("CREDIT_CARD", 20, 30)]) == (False, False)


def test_correct_kind_needs_an_entity_of_that_kind():
    findings = [Finding("DATE_TIME", 10, 20), Finding("CREDIT_CARD", 12, 18)]
    assert item_outcome("credit_debit_card", 10, 20, findings) == (True, True)


def test_kind_without_entities_is_never_correct():
    assert item_outcome("swift_bic", 0, 5, [Finding("CREDIT_CARD", 0, 5)]) == (True, False)


def test_empty_findings_are_dropped():
    assert find_entities([StubRecogniser([("CREDIT_CARD", 4, 4), ("DATE_TIME", 1, 3)])], "text") == [Finding("DATE_TIME", 1, 3)]


def document(text, spans, language="English"):
    return StructuredDocument(text, language, tuple(LabelledSpan(*span) for span in spans))


def test_counts_split_by_validity_and_language_and_skip_unscored_labels():
    text = "4111111111111111 and 4111111111111112 and Anna"
    doc = document(text, [(0, 16, "credit_card_number"), (21, 37, "credit_card_number"), (42, 46, "first_name")])
    recogniser = StubRecogniser([("CREDIT_CARD", 0, 16), ("DATE_TIME", 21, 30)])
    counts = count_documents("gretel", [doc], [recogniser])
    assert counts["English"] == counts["ALL"]
    assert counts["ALL"]["credit_debit_card"] == {
        "items": 2,
        "any_all": 2,
        "correct_all": 1,
        "valid_items": 1,
        "any_valid": 1,
        "correct_valid": 1,
    }
    assert set(counts["ALL"]) == {"credit_debit_card"}


def test_kind_without_checksum_has_equal_valid_and_all_counts():
    doc = document("a@example.com", [(0, 13, "email")])
    counts = count_documents("gretel", [doc], [StubRecogniser([])])["ALL"]["email"]
    assert counts["items"] == counts["valid_items"] == 1
    assert counts["any_all"] == counts["any_valid"] == 0


def test_merge_adds_counts():
    doc = document("a@example.com", [(0, 13, "email")])
    part = count_documents("gretel", [doc], [StubRecogniser([("EMAIL", 0, 5)])])
    total = {}
    merge_counts(total, part)
    merge_counts(total, part)
    assert total["ALL"]["email"]["items"] == 2


def test_summary_keeps_kinds_without_items_and_rates_none():
    summary = summarise({})
    assert summary["kinds"]["passport"]["items"] == 0
    assert summary["kinds"]["passport"]["any_rate_all"] is None
    assert set(summary["groups"]) == {"Government IDs", "Financial", "Travel and licences", "Contact and network"}


def test_result_holds_no_text_and_groups_sum_their_kinds():
    doc = document("a@example.com 1.2.3.4", [(0, 13, "email"), (14, 21, "ipv4")])
    result = build_result("gretel", "rev", ["Stub"], count_documents("gretel", [doc], [StubRecogniser([])]))
    assert result["overall"]["groups"]["Contact and network"]["items"] == 2
    assert "example.com" not in str(result)


def test_real_recognisers_find_a_synthetic_card_and_email_and_phone_in_any_region():
    text = "card 4111 1111 1111 1111, mail jane.doe@example.com, call +44 20 7946 0958."
    entities = {finding.entity for finding in find_entities(build_recognisers(), text)}
    assert {"CREDIT_CARD", "EMAIL_ADDRESS", "PHONE_NUMBER"} <= entities
