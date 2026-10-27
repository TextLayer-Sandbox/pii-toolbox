"""Stand-in detector, resolver, scorer and dataset shared by the evaluation run tests."""

from pii_toolbox.core import (
    AnnotatedDocument,
    AnswerSpan,
    Dataset,
    Detector,
    Document,
    EntityGroup,
    IdentityResolver,
    PiiType,
    ReidentificationRisk,
    Span,
    TabMetricsScorer,
    TabOfficialMetrics,
)
from pii_toolbox.detection.registry import FixedDetectorRegistry

CANARY = "ZXQCANARYTOKEN"
TEXT = f"Anna Testwell met Anna there. {CANARY} stays secret."
FULL_NAME = (0, 13)
SHORT_NAME = (18, 22)


class StubDetector(Detector):
    """Finds the full name only, in English documents."""

    name = "stub"
    languages = frozenset({"en"})
    detects = frozenset({PiiType.PERSON})

    def load_models(self) -> None:
        pass

    def detect(self, document):
        return [Span(*FULL_NAME, PiiType.PERSON, "PERSON", ("stub",), None)]


class StubResolver(IdentityResolver):
    name = "stub_resolver"

    def resolve(self, document, spans):
        added = Span(*SHORT_NAME, PiiType.PERSON, "PERSON", ("stub_resolver",), None)
        return [EntityGroup((spans[0], added), spans[0])]


class StubTabScorer(TabMetricsScorer):
    def score(self, results):
        return TabOfficialMetrics(0.5, 0.25, 0.75)


class StubDataset(Dataset):
    def __init__(self, name, data_dir=None):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def languages(self):
        return frozenset({"en", "fr"})

    def load(self, split, language=None):
        answers = (
            AnswerSpan(*FULL_NAME, PiiType.PERSON, "PERSON", "A", ReidentificationRisk.DIRECT, "e1"),
            AnswerSpan(*SHORT_NAME, PiiType.PERSON, "PERSON", "A", ReidentificationRisk.DIRECT, "e1"),
        )
        for doc_id, lang in (("d1", "en"), ("d2", "fr")):
            if language in (None, lang):
                yield AnnotatedDocument(Document(doc_id, TEXT, lang), answers)


def stub_registry():
    return FixedDetectorRegistry(factory_overrides={"stub": StubDetector})
