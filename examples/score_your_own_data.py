"""Runnable scoring example: measure a detector on a small hand-labelled dataset.

Run from the repository root with `uv run python examples/score_your_own_data.py`.
"""

from pii_toolbox.core import AnswerSpan, Document, PiiType, SpanMatchRule
from pii_toolbox.detection.presidio_structured import PresidioStructuredDetector
from pii_toolbox.evaluation.scorer import MatchingDetectionScorer

# Each sentence lists the exact strings that should be masked; spans are located from them.
LABELLED_TEXTS = (
    ("Write to jo.example@example.com about the invoice.", ("jo.example@example.com",)),
    ("Call +1 415-555-0123 or email sam@example.com.", ("+1 415-555-0123", "sam@example.com")),
    ("The docs are at www.example.com/help.", ("www.example.com/help",)),
    ("Alex Example lives nearby and has no contact details here.", ("Alex Example",)),
)
TYPE_OF_ANSWER = {"Alex Example": PiiType.PERSON}


def answers_for(text: str, values: tuple[str, ...]) -> list[AnswerSpan]:
    answers = []
    for value in values:
        start = text.index(value)
        pii_type = TYPE_OF_ANSWER.get(value, PiiType.IDENTIFIER)
        answers.append(AnswerSpan(start, start + len(value), pii_type, pii_type.value, None, None, None))
    return answers


def main() -> None:
    detector = PresidioStructuredDetector()
    results = []
    for number, (text, values) in enumerate(LABELLED_TEXTS, start=1):
        document = Document(f"example-{number}", text, "en")
        results.append((document, detector.detect(document), answers_for(text, values)))

    scorer = MatchingDetectionScorer()
    for label, rule in (("strict", SpanMatchRule.EXACT_BOUNDARIES), ("relaxed", SpanMatchRule.ANY_OVERLAP)):
        score = scorer.score(results, rule, reported_types=detector.detects).overall_micro
        print(f"{label}: precision={score.precision:.2f} recall={score.recall:.2f} f2={score.f2:.2f}")


if __name__ == "__main__":
    main()
