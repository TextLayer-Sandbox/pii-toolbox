"""The structured PII kinds scored in the structured evaluation: groups, dataset label maps, Presidio entity maps and validity checks."""

import re
from dataclasses import dataclass
from functools import cache

GOVERNMENT_IDS = "Government IDs"
FINANCIAL = "Financial"
TRAVEL_AND_LICENCES = "Travel and licences"
CONTACT_AND_NETWORK = "Contact and network"

GROUPS: dict[str, tuple[str, ...]] = {
    GOVERNMENT_IDS: ("ssn", "tax_id", "national_id"),
    FINANCIAL: ("credit_debit_card", "account_number", "bank_routing_number", "iban", "bban", "swift_bic"),
    TRAVEL_AND_LICENCES: ("passport", "driver_licence", "certificate_licence_number"),
    CONTACT_AND_NETWORK: ("email", "phone", "fax", "url", "ipv4", "ipv6", "mac_address"),
}

KINDS: tuple[str, ...] = tuple(kind for kinds in GROUPS.values() for kind in kinds)

KIND_GROUP: dict[str, str] = {kind: group for group, kinds in GROUPS.items() for kind in kinds}

NEMOTRON_LABEL_KINDS: dict[str, str] = {
    "ssn": "ssn",
    "tax_id": "tax_id",
    "national_id": "national_id",
    "credit_debit_card": "credit_debit_card",
    "account_number": "account_number",
    "bank_routing_number": "bank_routing_number",
    "swift_bic": "swift_bic",
    "certificate_license_number": "certificate_licence_number",
    "email": "email",
    "phone_number": "phone",
    "fax_number": "fax",
    "url": "url",
    "ipv4": "ipv4",
    "ipv6": "ipv6",
    "mac_address": "mac_address",
}

GRETEL_LABEL_KINDS: dict[str, str] = {
    "ssn": "ssn",
    "credit_card_number": "credit_debit_card",
    "bank_routing_number": "bank_routing_number",
    "iban": "iban",
    "bban": "bban",
    "swift_bic_code": "swift_bic",
    "passport_number": "passport",
    "driver_license_number": "driver_licence",
    "email": "email",
    "phone_number": "phone",
    "ipv4": "ipv4",
    "ipv6": "ipv6",
}

DATASET_LABEL_KINDS: dict[str, dict[str, str]] = {
    "nemotron": NEMOTRON_LABEL_KINDS,
    "gretel": GRETEL_LABEL_KINDS,
}

_SOCIAL_SECURITY_ENTITIES = frozenset(
    {
        "US_SSN",
        "CA_SIN",
        "UK_NINO",
        "DE_SOCIAL_SECURITY",
        "SE_PERSONNUMMER",
        "FI_PERSONAL_IDENTITY_CODE",
        "PL_PESEL",
        "KR_RRN",
    }
)

_NATIONAL_ID_ENTITIES = _SOCIAL_SECURITY_ENTITIES | frozenset(
    {
        "ES_NIF",
        "ES_NIE",
        "IT_IDENTITY_CARD",
        "IT_FISCAL_CODE",
        "DE_ID_CARD",
        "IN_AADHAAR",
        "ZA_ID_NUMBER",
        "TR_NATIONAL_ID",
        "NG_NIN",
        "TH_TNIN",
        "KR_FRN",
        "PH_UMID",
        "SG_NRIC_FIN",
    }
)

_TAX_ID_ENTITIES = frozenset(
    {
        "DE_TAX_ID",
        "DE_TAX_NUMBER",
        "DE_VAT_ID",
        "IT_VAT_CODE",
        "IN_PAN",
        "IN_GSTIN",
        "PH_TIN",
        "AU_TFN",
        "AU_ABN",
        "US_ITIN",
    }
)

_PASSPORT_ENTITIES = frozenset(
    {"US_PASSPORT", "UK_PASSPORT", "IN_PASSPORT", "ES_PASSPORT", "IT_PASSPORT", "KR_PASSPORT", "DE_PASSPORT"}
)

_DRIVER_LICENCE_ENTITIES = frozenset(
    {"US_DRIVER_LICENSE", "UK_DRIVING_LICENCE", "IT_DRIVER_LICENSE", "KR_DRIVER_LICENSE", "DE_FUEHRERSCHEIN"}
)

KIND_ENTITIES: dict[str, frozenset[str]] = {
    "ssn": _SOCIAL_SECURITY_ENTITIES,
    "tax_id": _TAX_ID_ENTITIES,
    "national_id": _NATIONAL_ID_ENTITIES,
    "credit_debit_card": frozenset({"CREDIT_CARD"}),
    "account_number": frozenset({"US_BANK_NUMBER"}),
    "bank_routing_number": frozenset({"ABA_ROUTING_NUMBER"}),
    "iban": frozenset({"IBAN_CODE"}),
    "bban": frozenset(),
    "swift_bic": frozenset(),
    "passport": _PASSPORT_ENTITIES,
    "driver_licence": _DRIVER_LICENCE_ENTITIES,
    "certificate_licence_number": _DRIVER_LICENCE_ENTITIES | {"MEDICAL_LICENSE"},
    "email": frozenset({"EMAIL_ADDRESS"}),
    "phone": frozenset({"PHONE_NUMBER"}),
    "fax": frozenset({"PHONE_NUMBER"}),
    "url": frozenset({"URL"}),
    "ipv4": frozenset({"IP_ADDRESS"}),
    "ipv6": frozenset({"IP_ADDRESS"}),
    "mac_address": frozenset({"MAC_ADDRESS"}),
}


def kind_of_label(dataset: str, label: str) -> str | None:
    """The kind a dataset label belongs to, None for a label that is not scored."""
    return DATASET_LABEL_KINDS[dataset].get(label)


def luhn_valid(value: str) -> bool:
    """True for a 12 to 19 digit number, spaces and dashes allowed, whose Luhn checksum holds."""
    digits = re.sub(r"[ -]", "", value)
    if not digits.isascii() or not digits.isdigit() or not 12 <= len(digits) <= 19:
        return False
    total = 0
    for position, character in enumerate(reversed(digits)):
        digit = int(character)
        if position % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def iban_valid(value: str) -> bool:
    """True for an IBAN whose ISO 7064 mod-97 remainder is 1, spaces allowed."""
    compact = value.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(character, 36)) for character in rearranged)
    return int(number) % 97 == 1


@cache
def _ssn_recogniser():
    from presidio_analyzer.predefined_recognizers import UsSsnRecognizer

    return UsSsnRecognizer()


@cache
def _routing_recogniser():
    from presidio_analyzer.predefined_recognizers import AbaRoutingRecognizer

    return AbaRoutingRecognizer()


def us_ssn_valid(value: str) -> bool:
    """True for nine digits, with one kind of delimiter at most, that Presidio's SSN rules do not rule out."""
    if not re.fullmatch(r"\d{3}([ .-]?)\d{2}\1\d{4}", value, re.ASCII):
        return False
    return not _ssn_recogniser().invalidate_result(value)


def routing_number_valid(value: str) -> bool:
    """True for nine digits, dashes allowed, that pass the ABA routing checksum."""
    digits = value.replace("-", "")
    if not digits.isascii() or not digits.isdigit() or len(digits) != 9:
        return False
    return _routing_recogniser().validate_result(value)


VALIDITY_CHECKS = {
    "credit_debit_card": luhn_valid,
    "iban": iban_valid,
    "ssn": us_ssn_valid,
    "bank_routing_number": routing_number_valid,
}


def is_validly_formed(kind: str, value: str) -> bool:
    """Whether a labelled value passes its kind's checksum or validity rules; kinds without one are always valid."""
    check = VALIDITY_CHECKS.get(kind)
    if check is None:
        return True
    return check(value)
