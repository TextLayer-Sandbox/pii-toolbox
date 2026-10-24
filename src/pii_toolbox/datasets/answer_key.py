"""Answer key that merges annotators: a span is PII if any annotator marked it DIRECT or QUASI."""

from collections.abc import Sequence
from dataclasses import replace

from pii_toolbox.core import AnswerKeyBuilder, AnswerSpan, ReidentificationRisk

_RISK_STRENGTH = {ReidentificationRisk.QUASI: 1, ReidentificationRisk.DIRECT: 2}


class AnyAnnotatorAnswerKeyBuilder(AnswerKeyBuilder):
    """Keeps every span some annotator marked DIRECT or QUASI, once per distinct boundary and label.

    Spans with different boundaries stay separate. Identical spans from several annotators
    merge into one that takes the stronger risk and no single annotator. Answers whose risk
    is None come from datasets without risk grading and are all kept.
    """

    def build(self, answers: Sequence[AnswerSpan]) -> tuple[AnswerSpan, ...]:
        merged: dict[tuple[int, int, str], AnswerSpan] = {}
        for answer in answers:
            if answer.risk is ReidentificationRisk.NONE:
                continue
            key = (answer.start, answer.end, answer.label)
            existing = merged.get(key)
            if existing is None:
                merged[key] = answer
                continue
            if _RISK_STRENGTH.get(answer.risk, 0) > _RISK_STRENGTH.get(existing.risk, 0):
                existing = replace(existing, risk=answer.risk)
            merged[key] = replace(existing, annotator=None) if existing.annotator != answer.annotator else existing
        return tuple(sorted(merged.values(), key=lambda span: (span.start, span.end, span.label)))
