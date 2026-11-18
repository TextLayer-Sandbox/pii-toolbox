"""Detector for structured identifiers using Presidio's pattern and checksum recognisers.

Only the recognisers listed here run, and only when this detector is part of a
configuration; Presidio's name-finding recognisers and its spaCy engine are never involved,
so no model files are read and nothing touches the network.
"""

from pii_toolbox.core import PRESIDIO_LABELS, Detector, Document, PiiType, Span

ENTITY_TYPES = (
    "CREDIT_CARD",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    "URL",
    "PHONE_NUMBER",
    "US_SSN",
)


def build_offline_email_recogniser():
    """An email recogniser that validates domains against tldextract's bundled suffix list."""
    from presidio_analyzer.predefined_recognizers import EmailRecognizer
    from tldextract import TLDExtract

    # An empty suffix-list URL tuple makes tldextract use its bundled list instead of downloading one.
    offline_domain_parser = TLDExtract(suffix_list_urls=(), cache_dir=None)

    class OfflineEmailRecognizer(EmailRecognizer):
        def validate_result(self, pattern_text: str) -> bool:
            return offline_domain_parser(pattern_text).fqdn != ""

    return OfflineEmailRecognizer()


def _build_recognisers() -> list:
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer,
        IbanRecognizer,
        IpRecognizer,
        PhoneRecognizer,
        UrlRecognizer,
        UsSsnRecognizer,
    )

    return [
        CreditCardRecognizer(),
        build_offline_email_recogniser(),
        IbanRecognizer(),
        IpRecognizer(),
        UrlRecognizer(),
        PhoneRecognizer(),
        UsSsnRecognizer(),
    ]


class PresidioStructuredDetector(Detector):
    """Finds card numbers, emails, IBANs, IP addresses, URLs, phone numbers and US SSNs."""

    def __init__(self) -> None:
        self._recognisers: list | None = None

    @property
    def name(self) -> str:
        return "presidio_structured"

    @property
    def languages(self) -> frozenset[str] | None:
        return None

    @property
    def detects(self) -> frozenset[PiiType]:
        return frozenset({PiiType.IDENTIFIER})

    def load_models(self) -> None:
        self._recognisers = _build_recognisers()

    def detect(self, document: Document) -> list[Span]:
        if self._recognisers is None:
            self.load_models()
        spans = [
            Span(
                start=result.start,
                end=result.end,
                pii_type=PRESIDIO_LABELS[result.entity_type],
                detector_label=result.entity_type,
                source=(self.name,),
                score=result.score,
            )
            for recogniser in self._recognisers
            for result in recogniser.analyze(document.text, list(recogniser.supported_entities))
            if result.entity_type in ENTITY_TYPES and result.start < result.end
        ]
        return _without_contained_spans(spans)


def _without_contained_spans(spans: list[Span]) -> list[Span]:
    """Drops spans lying inside a wider one, such as the host name Presidio finds inside an email address."""
    kept = [
        span
        for span in spans
        if not any(
            other is not span
            and other.start <= span.start
            and span.end <= other.end
            and (other.end - other.start > span.end - span.start or other.score > span.score)
            for other in spans
        )
    ]
    return sorted(kept, key=lambda span: (span.start, span.end))
