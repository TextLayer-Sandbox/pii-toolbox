"""Translations from each dataset's and tool's own labels to the common type set.

A value of None marks a label with no common counterpart.
"""

from collections.abc import Mapping

from pii_toolbox.core.types import PiiType

OptionalLabelMap = Mapping[str, PiiType | None]

TAB_LABELS: OptionalLabelMap = {
    "PERSON": PiiType.PERSON,
    "ORG": PiiType.ORGANIZATION,
    "LOC": PiiType.LOCATION,
    "DATETIME": PiiType.DATE_TIME,
    "QUANTITY": PiiType.QUANTITY,
    "CODE": PiiType.IDENTIFIER,
    "DEM": None,
    "MISC": None,
}

# DATE and TIME merge because the other dataset has a single date-time type.
MAPA_LABELS: OptionalLabelMap = {
    "PERSON": PiiType.PERSON,
    "ORGANISATION": PiiType.ORGANIZATION,
    "ADDRESS": PiiType.LOCATION,
    "DATE": PiiType.DATE_TIME,
    "TIME": PiiType.DATE_TIME,
    "AMOUNT": PiiType.QUANTITY,
}

PRESIDIO_LABELS: OptionalLabelMap = {
    "PERSON": PiiType.PERSON,
    "ORGANIZATION": PiiType.ORGANIZATION,
    "LOCATION": PiiType.LOCATION,
    "DATE_TIME": PiiType.DATE_TIME,
    "EMAIL_ADDRESS": PiiType.IDENTIFIER,
    "PHONE_NUMBER": PiiType.IDENTIFIER,
    "CREDIT_CARD": PiiType.IDENTIFIER,
    "IBAN_CODE": PiiType.IDENTIFIER,
    "IP_ADDRESS": PiiType.IDENTIFIER,
    "URL": PiiType.IDENTIFIER,
    "US_SSN": PiiType.IDENTIFIER,
}

GLINER_LABELS: OptionalLabelMap = {
    "person": PiiType.PERSON,
    "name": PiiType.PERSON,
    "organization": PiiType.ORGANIZATION,
    "location": PiiType.LOCATION,
    "address": PiiType.LOCATION,
}

ROBERTA_LABELS: OptionalLabelMap = {
    "PER": PiiType.PERSON,
    "ORG": PiiType.ORGANIZATION,
    "LOC": PiiType.LOCATION,
    "MISC": None,
}


def to_pii_type(label_map: OptionalLabelMap, label: str) -> PiiType | None:
    """Returns the common type for a label, None when it has none; raises KeyError naming an unknown label."""
    if label not in label_map:
        raise KeyError(label)
    return label_map[label]
