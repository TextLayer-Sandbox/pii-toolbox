"""Runs one detector configuration over one dataset split and scores it into an EvaluationRun.

Detectors come from an injected registry, the optional co-reference resolver adds name forms
the detector missed, and the dataset answer key is the any-annotator key. Only aggregate
scores leave this module; documents and their predictions stay in memory.
"""

from dataclasses import dataclass

from pii_toolbox.core import (
    AnnotatedDocument,
    Dataset,
    DatasetSplit,
    Detector,
    DetectorRegistry,
    EvaluationRun,
    IdentityResolver,
    RunEnvironment,
    Span,
    SpanMatchRule,
    TabMetricsScorer,
)
from pii_toolbox.datasets.answer_key import AnyAnnotatorAnswerKeyBuilder
from pii_toolbox.detection.registry import build_from_config
from pii_toolbox.evaluation.environment import collect_environment
from pii_toolbox.evaluation.results_io import ResultRecord
from pii_toolbox.evaluation.scorer import MatchingDetectionScorer

TAB_DATASET_NAME = "tab"
COREFERENCE_LANGUAGE = "en"


@dataclass(frozen=True)
class RunRequest:
    """What to run: a configuration entry (`a` or `a+b`), a dataset, a split and whether to resolve co-reference."""

    tool: str
    dataset: Dataset
    split: DatasetSplit
    coreference: bool
    language: str | None = None


def _with_resolved_names(resolver: IdentityResolver, annotated: AnnotatedDocument, spans: list[Span]) -> list[Span]:
    """Adds the spans of every group member the detector did not report."""
    reported = {(span.start, span.end) for span in spans}
    added = []
    for group in resolver.resolve(annotated.document, spans):
        for member in group.spans:
            if (member.start, member.end) not in reported:
                reported.add((member.start, member.end))
                added.append(member)
    return spans + added


def _is_scored(detector: Detector, request: RunRequest, annotated: AnnotatedDocument) -> bool:
    language = annotated.document.language
    if detector.languages is not None and language not in detector.languages:
        return False
    return not request.coreference or language == COREFERENCE_LANGUAGE


def run_evaluation(
    request: RunRequest,
    registry: DetectorRegistry,
    tab_scorer: TabMetricsScorer | None = None,
    resolver: IdentityResolver | None = None,
    environment: RunEnvironment | None = None,
) -> ResultRecord:
    """Scores the configuration on the split and records what the detector supports.

    Documents in a language the detector does not support are left out, and so are non-English
    documents when co-reference is on. `tab_scorer` is needed only for the TAB dataset.
    """
    if request.coreference and resolver is None:
        raise ValueError("co-reference is on but no resolver was given")
    detector = build_from_config([request.tool], registry)[0]
    detector.load_models()
    answer_keys = AnyAnnotatorAnswerKeyBuilder()
    scored: list[tuple[AnnotatedDocument, list[Span]]] = []
    for annotated in request.dataset.load(request.split, request.language):
        if not _is_scored(detector, request, annotated):
            continue
        spans = detector.detect(annotated.document)
        if request.coreference:
            spans = _with_resolved_names(resolver, annotated, spans)
        scored.append((annotated, spans))
    results = [
        (annotated.document, spans, answer_keys.build(annotated.answers)) for annotated, spans in scored
    ]
    scorer = MatchingDetectionScorer()
    is_tab = request.dataset.name.lower() == TAB_DATASET_NAME
    if is_tab and tab_scorer is None:
        raise ValueError("the TAB dataset needs a TAB metrics scorer")
    run = EvaluationRun(
        tool=request.tool,
        coreference=request.coreference,
        dataset=request.dataset.name.lower(),
        split=request.split,
        exact_boundaries=scorer.score(
            results,
            SpanMatchRule.EXACT_BOUNDARIES,
            reported_types=detector.detects,
            annotated_types=request.dataset.annotated_types,
        ),
        any_overlap=scorer.score(
            results,
            SpanMatchRule.ANY_OVERLAP,
            reported_types=detector.detects,
            annotated_types=request.dataset.annotated_types,
        ),
        tab_metrics=tab_scorer.score(scored) if is_tab else None,
        environment=environment or collect_environment(),
    )
    return ResultRecord(run, detector.languages, detector.detects)
