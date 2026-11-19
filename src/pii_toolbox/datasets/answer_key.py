"""Answer key that merges annotators: a span is kept if any annotator did not mark it NONE."""

from collections.abc import Sequence
from dataclasses import replace

from pii_toolbox.core import AnswerKeyBuilder, AnswerSpan, ReidentificationRisk

_RISK_STRENGTH: dict[ReidentificationRisk | None, int] = {ReidentificationRisk.QUASI: 1, ReidentificationRisk.DIRECT: 2}


class AnyAnnotatorAnswerKeyBuilder(AnswerKeyBuilder):
    """Keeps every answer not marked NONE, merging identical spans of several annotators into one with the stronger risk and no annotator."""

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
