import pytest

from pii_toolbox.evaluation.structured_kinds import (
    GRETEL_LABEL_KINDS,
    GROUPS,
    KIND_ENTITIES,
    KINDS,
    NEMOTRON_LABEL_KINDS,
    iban_valid,
    is_validly_formed,
    kind_of_label,
    luhn_valid,
    routing_number_valid,
    us_ssn_valid,
)


def test_every_kind_is_in_exactly_one_group_and_has_an_entity_set():
    grouped = [kind for kinds in GROUPS.values() for kind in kinds]
    assert sorted(grouped) == sorted(set(grouped))
    assert set(KIND_ENTITIES) == set(KINDS)


def test_every_label_maps_to_a_known_kind():
    assert set(NEMOTRON_LABEL_KINDS.values()) <= set(KINDS)
    assert set(GRETEL_LABEL_KINDS.values()) <= set(KINDS)


def test_unscored_label_has_no_kind():
    assert kind_of_label("nemotron", "first_name") is None
    assert kind_of_label("gretel", "credit_card_security_code") is None


def test_labels_map_to_kinds_per_dataset():
    assert kind_of_label("nemotron", "fax_number") == "fax"
    assert kind_of_label("gretel", "swift_bic_code") == "swift_bic"
    assert kind_of_label("gretel", "credit_card_number") == "credit_debit_card"


def test_kinds_without_a_recogniser_have_no_entities():
    assert KIND_ENTITIES["bban"] == frozenset()
    assert KIND_ENTITIES["swift_bic"] == frozenset()


def test_every_mapped_entity_is_produced_by_an_installed_recogniser():
    from pii_toolbox.evaluation.structured import build_recognisers

    produced = {entity for recogniser in build_recognisers() for entity in recogniser.supported_entities}
    mapped = set().union(*KIND_ENTITIES.values())
    assert mapped <= produced


@pytest.mark.parametrize("value, expected", [("4111 1111 1111 1111", True), ("4111-1111-1111-1111", True), ("4111111111111112", False), ("4111 abcd", False)])
def test_luhn(value, expected):
    assert luhn_valid(value) is expected


@pytest.mark.parametrize("value, expected", [("GB82 WEST 1234 5698 7654 32", True), ("GB82WEST12345698765433", False), ("not an iban", False)])
def test_iban_mod97(value, expected):
    assert iban_valid(value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [("457-55-5462", True), ("457555462", True), ("666-12-3456", False), ("123-45-6789", False), ("457-55 5462", False), ("12-345-6789", False)],
)
def test_us_ssn_rules(value, expected):
    assert us_ssn_valid(value) is expected


@pytest.mark.parametrize("value, expected", [("021000021", True), ("021000022", False), ("12345", False)])
def test_routing_checksum(value, expected):
    assert routing_number_valid(value) is expected


def test_kind_without_a_checksum_is_always_valid():
    assert is_validly_formed("email", "anything at all") is True


def test_kind_with_a_checksum_uses_it():
    assert is_validly_formed("credit_debit_card", "4111111111111112") is False
