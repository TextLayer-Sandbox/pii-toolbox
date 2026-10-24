from pii_toolbox.core import Document, PiiType
from pii_toolbox.detection.presidio_structured import PresidioStructuredDetector

TEXT = (
    "Card 4111 1111 1111 1111 and bad card 4111 1111 1111 1112. "
    "Write to jane.doe@example.com. IBAN GB82 WEST 1234 5698 7654 32, host 192.0.2.17, "
    "page https://example.com/a phone +44 20 7946 0958, ssn 078-05-1120, us ssn 536-90-4399."
)


def found(spans):
    return {TEXT[s.start : s.end]: s for s in spans}


def run():
    detector = PresidioStructuredDetector()
    detector.load_models()
    return detector.detect(Document(id="d", text=TEXT, language="en"))


def test_flags_each_structured_identifier_with_presidio_label():
    spans = found(run())
    expected = {
        "4111 1111 1111 1111": "CREDIT_CARD",
        "jane.doe@example.com": "EMAIL_ADDRESS",
        "GB82 WEST 1234 5698 7654 32": "IBAN_CODE",
        "192.0.2.17": "IP_ADDRESS",
        "https://example.com/a": "URL",
        "+44 20 7946 0958": "PHONE_NUMBER",
    }
    for value, label in expected.items():
        assert spans[value].detector_label == label
    assert "078-05-1120" in spans
    assert spans["536-90-4399"].detector_label == "US_SSN"


def test_card_with_bad_check_digit_is_not_flagged():
    assert "4111 1111 1111 1112" not in found(run())


def test_every_span_is_identifier_inside_the_text():
    spans = run()
    assert spans
    for span in spans:
        assert span.pii_type is PiiType.IDENTIFIER
        assert 0 <= span.start < span.end <= len(TEXT)
        assert span.source == ("presidio_structured",)


def test_fragments_inside_a_wider_span_are_dropped():
    values = list(found(run()))
    assert "jane.do" not in values
    assert "example.com" not in values


def test_interface_properties():
    detector = PresidioStructuredDetector()
    assert detector.name == "presidio_structured"
    assert detector.languages is None
    assert detector.detects == frozenset({PiiType.IDENTIFIER})
