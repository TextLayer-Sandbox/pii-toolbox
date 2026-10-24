import pytest

from pii_toolbox.core import PiiType
from pii_toolbox.core.label_maps import (
    GLINER_LABELS,
    MAPA_LABELS,
    PRESIDIO_LABELS,
    ROBERTA_LABELS,
    TAB_LABELS,
    to_pii_type,
)

P, O, L, D, Q, I = (
    PiiType.PERSON,
    PiiType.ORGANIZATION,
    PiiType.LOCATION,
    PiiType.DATE_TIME,
    PiiType.QUANTITY,
    PiiType.IDENTIFIER,
)

ROWS = [
    (TAB_LABELS, "PERSON", P),
    (TAB_LABELS, "ORG", O),
    (TAB_LABELS, "LOC", L),
    (TAB_LABELS, "DATETIME", D),
    (TAB_LABELS, "QUANTITY", Q),
    (TAB_LABELS, "CODE", I),
    (TAB_LABELS, "DEM", None),
    (TAB_LABELS, "MISC", None),
    (MAPA_LABELS, "PERSON", P),
    (MAPA_LABELS, "ORGANISATION", O),
    (MAPA_LABELS, "ADDRESS", L),
    (MAPA_LABELS, "DATE", D),
    (MAPA_LABELS, "TIME", D),
    (MAPA_LABELS, "AMOUNT", Q),
    (PRESIDIO_LABELS, "PERSON", P),
    (PRESIDIO_LABELS, "ORGANIZATION", O),
    (PRESIDIO_LABELS, "LOCATION", L),
    (PRESIDIO_LABELS, "DATE_TIME", D),
    (PRESIDIO_LABELS, "EMAIL_ADDRESS", I),
    (PRESIDIO_LABELS, "PHONE_NUMBER", I),
    (PRESIDIO_LABELS, "CREDIT_CARD", I),
    (PRESIDIO_LABELS, "IBAN_CODE", I),
    (PRESIDIO_LABELS, "IP_ADDRESS", I),
    (PRESIDIO_LABELS, "URL", I),
    (PRESIDIO_LABELS, "US_SSN", I),
    (GLINER_LABELS, "person", P),
    (GLINER_LABELS, "name", P),
    (GLINER_LABELS, "organization", O),
    (GLINER_LABELS, "location", L),
    (GLINER_LABELS, "address", L),
    (ROBERTA_LABELS, "PER", P),
    (ROBERTA_LABELS, "ORG", O),
    (ROBERTA_LABELS, "LOC", L),
    (ROBERTA_LABELS, "MISC", None),
]


@pytest.mark.parametrize("label_map, label, expected", ROWS)
def test_label_maps_to_common_type(label_map, label, expected):
    assert to_pii_type(label_map, label) is expected


def test_unknown_label_raises_key_error_naming_the_label():
    with pytest.raises(KeyError, match="NOT_A_LABEL"):
        to_pii_type(TAB_LABELS, "NOT_A_LABEL")
