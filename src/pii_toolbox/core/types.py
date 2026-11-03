"""Foundational types and interfaces for local PII detection, masking and evaluation.

Character positions are offsets into the original text: start inclusive, end exclusive.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Self

# ---------------------------------------------------------------------------
# Core data
# ---------------------------------------------------------------------------


class PiiType(Enum):
    """Common type set that every detector and dataset is mapped onto for scoring."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE_TIME = "DATE_TIME"
    QUANTITY = "QUANTITY"
    # Structured codes (email, phone, card, IBAN, national and tax IDs, IP, URL) kept as
    # one type because the answer keys do not distinguish them.
    IDENTIFIER = "IDENTIFIER"


LabelMap = Mapping[str, PiiType]
"""Translates a tool's or dataset's own labels into the common type set."""


@dataclass(frozen=True, slots=True)
class Span:
    """A detected region of text. Invariant: 0 <= start < end <= len(text)."""

    start: int
    end: int
    pii_type: PiiType  # common type, used for scoring
    # The detector's own detailed name (e.g. EMAIL_ADDRESS, IBAN_CODE, both IDENTIFIER);
    # used in placeholders so text read downstream still says what kind of value was masked.
    detector_label: str
    source: tuple[str, ...]  # names of every detector that contributed; more than one after a merge
    score: float | None  # detector confidence, None when the detector gives none

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"invalid span bounds: start={self.start}, end={self.end}")


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    language: str  # ISO 639-1 code


class ReidentificationRisk(Enum):
    """How much a mention gives away about who a person is, as marked by dataset annotators.

    Needed because recall on direct and on quasi identifiers is reported separately.
    """

    DIRECT = "DIRECT"  # identifies a person on its own (e.g. a full name)
    QUASI = "QUASI"  # identifies a person in combination with other details (e.g. age plus occupation plus town)
    NONE = "NONE"  # needs no masking


@dataclass(frozen=True, slots=True)
class AnswerSpan:
    """A correct answer from a dataset, in the dataset's own terms."""

    start: int
    end: int
    pii_type: PiiType | None  # None for dataset types with no common counterpart
    label: str  # the dataset's own label
    annotator: str | None
    risk: ReidentificationRisk | None
    entity_id: str | None  # ties mentions of one entity together


@dataclass(frozen=True, slots=True)
class AnnotatedDocument:
    document: Document
    answers: tuple[AnswerSpan, ...]


@dataclass(frozen=True, slots=True)
class EntityGroup:
    """Spans that refer to the same person. Holds positions only, never text.

    A group may hold a span the detector did not find; such a span's `source` names the
    resolver. A group always holds at least one detected name.
    """

    spans: tuple[Span, ...]
    canonical: Span  # the fullest form of the name; one of `spans`


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class Detector(ABC):
    """Finds PII in text."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def languages(self) -> frozenset[str] | None:
        """ISO codes this detector supports; None means any language."""
        ...

    @property
    @abstractmethod
    def detects(self) -> frozenset[PiiType]:
        """The common types this configured detector can report.

        Types outside it count as misses in the overall score and are marked not applicable
        in the per-type breakdown.
        """
        ...

    @abstractmethod
    def load_models(self) -> None:
        """Reads model files into memory and starts any runtime or worker process needed.

        Separate from detection so load time is measured apart from per-call time;
        does nothing for detectors without models.
        """
        ...

    @abstractmethod
    def detect(self, document: Document) -> list[Span]: ...


class DetectorUnion(Detector):
    """Runs several detectors and returns the union of their spans.

    Overlapping spans merge into the widest; the merged span takes the type and detector
    label of the longest contributing span, ties broken by the higher score, and its score
    is the highest among contributors.
    """

    @property
    @abstractmethod
    def members(self) -> tuple[Detector, ...]: ...


class DetectorRegistry(ABC):
    """The fixed list of detector names known to the program, each with how to build it.

    Selection by configuration is checked against it at startup.
    """

    @abstractmethod
    def names(self) -> frozenset[str]: ...

    @abstractmethod
    def build(self, name: str) -> Detector: ...


@dataclass(frozen=True, slots=True)
class TextSegment:
    offset: int  # position of the segment within the full text
    text: str


class TextSegmenter(ABC):
    """Cuts long text into overlapping segments so models with a limited input length can read it."""

    @abstractmethod
    def split(self, text: str) -> list[TextSegment]:
        """Returns overlapping segments that together cover the whole text."""
        ...


class IdentityResolver(ABC):
    """Groups spans that refer to the same person."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def resolve(self, document: Document, spans: Sequence[Span]) -> list[EntityGroup]:
        """Never returns a group that contains no detected name."""
        ...


# ---------------------------------------------------------------------------
# Session: swapping PII out and back in
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Placeholder:
    label: str  # detailed type label, e.g. EMAIL_ADDRESS
    number: int  # count of this label within the session, starting at 1


@dataclass(frozen=True, slots=True)
class PlaceholderToken:
    """A placeholder-shaped token found in text, with the placeholder it reads as."""

    start: int
    end: int
    placeholder: Placeholder


class PlaceholderFormat(ABC):
    """The single definition of what a placeholder looks like in text, shared by masking and restoring."""

    @abstractmethod
    def render(self, placeholder: Placeholder) -> str:
        """The exact text written into masked output, such as <PERSON_1>."""
        ...

    @abstractmethod
    def parse(self, text: str) -> list[PlaceholderToken]:
        """Finds whole tokens that read as placeholders, tolerating dropped brackets and changed case."""
        ...


@dataclass(frozen=True, slots=True)
class MaskResult:
    masked_text: str
    minted: tuple[Placeholder, ...]  # placeholders created by this call only
    placeholder_by_span: Mapping[Span, Placeholder] = field(default_factory=dict)  # every replaced span, including spans a resolver added


@dataclass(frozen=True, slots=True)
class RestoreResult:
    restored_text: str
    leftovers: tuple[str, ...]  # tokens that read as a placeholder with a label this session has minted but did not resolve; look-alikes with unknown labels are not reported


class PlaceholderStore(ABC):
    """Holds, per session, the original value each placeholder stands for.

    Originals are never written anywhere other than the store itself.
    """

    @abstractmethod
    def get_or_mint(self, session_id: str, label: str, original: str) -> tuple[Placeholder, bool]:
        """Returns the existing placeholder for the original, or mints the next-numbered one for the label.

        The bool is True when the placeholder was minted by this call.
        """
        ...

    @abstractmethod
    def alias(self, session_id: str, original: str, placeholder: Placeholder) -> None:
        """Makes another original resolve to an existing placeholder, so different forms of one name share it."""
        ...

    @abstractmethod
    def original(self, session_id: str, placeholder: Placeholder) -> str | None:
        """Returns the stored value; when aliases exist, this is the fullest form."""
        ...

    @abstractmethod
    def labels(self, session_id: str) -> frozenset[str]:
        """Every label that has been minted in the session."""
        ...

    @abstractmethod
    def discard(self, session_id: str) -> None:
        """Deletes every mapping held for the session."""
        ...


class RedactionSession(ABC):
    """One case's masking context. The same original always gets the same placeholder.

    A session is built from a detector, an optional resolver, a placeholder store, a placeholder format and a language.
    """

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def language(self) -> str:
        """ISO 639-1 code given when the session is opened; every text masked in it is taken to be in this language."""
        ...

    @property
    @abstractmethod
    def store(self) -> PlaceholderStore: ...

    @property
    @abstractmethod
    def format(self) -> PlaceholderFormat: ...

    @property
    @abstractmethod
    def detector(self) -> Detector: ...

    @property
    @abstractmethod
    def resolver(self) -> IdentityResolver | None:
        """None means name forms are not linked."""
        ...

    @abstractmethod
    def mask(self, text: str) -> MaskResult: ...

    @abstractmethod
    def restore(self, text: str) -> RestoreResult:
        """Swaps back only placeholders that exist in this session."""
        ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Closes the session whether or not the block raised."""
        ...


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class DatasetSplit(Enum):
    TRAIN = "TRAIN"
    DEV = "DEV"
    TEST = "TEST"


class Dataset(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def languages(self) -> frozenset[str]: ...

    @abstractmethod
    def load(
        self, split: DatasetSplit, language: str | None = None
    ) -> Iterator[AnnotatedDocument]:
        """Yields documents of the split; language None means every language."""
        ...


class SpanMatchRule(Enum):
    EXACT_BOUNDARIES = "EXACT_BOUNDARIES"  # same start and end
    ANY_OVERLAP = "ANY_OVERLAP"  # any overlap


@dataclass(frozen=True, slots=True)
class MatchCounts:
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True, slots=True)
class DetectionScore:
    precision: float
    recall: float
    f2: float  # F-beta with beta = 2, weighting recall over precision


@dataclass(frozen=True, slots=True)
class DetectionReport:
    rule: SpanMatchRule
    overall_micro: DetectionScore  # counts pooled over all types
    macro: DetectionScore  # plain average of the per-type scores; averages only the reported types
    by_type: Mapping[PiiType, DetectionScore | None]  # None marks a type the detector does not report
    by_language: Mapping[str, DetectionScore]


class SpanMatcher(ABC):
    @abstractmethod
    def match(
        self,
        predicted: Sequence[Span],
        answers: Sequence[AnswerSpan],
        rule: SpanMatchRule,
    ) -> Mapping[PiiType, MatchCounts]:
        """Greedy and one-to-one: each answer is credited to at most one prediction."""
        ...


class DetectionScorer(ABC):
    @abstractmethod
    def score(
        self,
        results: Iterable[tuple[Document, Sequence[Span], Sequence[AnswerSpan]]],
        rule: SpanMatchRule,
    ) -> DetectionReport:
        """Each item is a document with its predicted spans and its answers."""
        ...


class AnswerKeyBuilder(ABC):
    @abstractmethod
    def build(self, answers: Sequence[AnswerSpan]) -> tuple[AnswerSpan, ...]:
        """Collapses several annotators into one answer key: a span is PII if any annotator marked it DIRECT or QUASI; distinct boundaries stay distinct spans."""
        ...


@dataclass(frozen=True, slots=True)
class TabOfficialMetrics:
    er_di: float  # entity-level recall on direct identifiers
    er_qi: float  # entity-level recall on quasi-identifiers
    wp: float  # token-level weighted precision


class TabMetricsScorer(ABC):
    @abstractmethod
    def score(
        self, results: Iterable[tuple[AnnotatedDocument, Sequence[Span]]]
    ) -> TabOfficialMetrics:
        """Computes TAB's own metrics by its definitions, micro-averaged over annotators."""
        ...


@dataclass(frozen=True, slots=True)
class CostMeasurement:
    peak_memory_bytes: int
    load_seconds: float
    throughput_chars_per_second: Mapping[int, float]  # keyed by input size in characters
    latency_per_1000_chars_median_seconds: float


class CostRunner(ABC):
    @abstractmethod
    def measure(
        self, detector_name: str, input_sizes: Sequence[int]
    ) -> CostMeasurement:
        """Runs in a fresh process so memory figures belong to that detector alone."""
        ...


@dataclass(frozen=True, slots=True)
class RunEnvironment:
    machine: str
    os: str
    runtime: str
    library_versions: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    tool: str
    coreference: bool
    dataset: str
    split: DatasetSplit
    exact_boundaries: DetectionReport
    any_overlap: DetectionReport
    tab_metrics: TabOfficialMetrics | None  # only present for the TAB dataset
    environment: RunEnvironment
