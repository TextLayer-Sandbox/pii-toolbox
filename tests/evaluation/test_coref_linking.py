from pii_toolbox.core import (
    AnnotatedDocument,
    AnswerSpan,
    Detector,
    Document,
    PersonMentions,
    IdentityResolver,
    PiiType,
    Placeholder,
    ReidentificationRisk,
    Span,
)
from pii_toolbox.evaluation.coref_linking import LinkingCounts, count_linking, linking_report, measure_linking

TEXT = "Anna Smith met Annie. Anna Smith met Bob Jones. Bob left. Bob left."


def span(start: int, end: int) -> Span:
    return Span(start, end, PiiType.PERSON, "PERSON", ("test",), None)


def answer(start: int, end: int, annotator: str, entity_id: str, risk=ReidentificationRisk.DIRECT) -> AnswerSpan:
    return AnswerSpan(start, end, PiiType.PERSON, "PERSON", annotator, risk, entity_id)


def position(text: str, occurrence: int = 0) -> tuple[int, int]:
    start = -1
    for _ in range(occurrence + 1):
        start = TEXT.index(text, start + 1)
    return start, start + len(text)


DOCUMENT = Document("doc", TEXT, "en")
ANNA_FULL, ANNA_SHORT, ANNA_AGAIN = position("Anna Smith"), position("Annie"), position("Anna Smith", 1)
BOB_FULL, BOB_FIRST, BOB_SECOND = position("Bob Jones"), position("Bob", 1), position("Bob", 2)
STRANGER = position("met", 0)


def gold(annotator: str):
    return (
        answer(*ANNA_FULL, annotator, "a_e1"),
        answer(*ANNA_SHORT, annotator, "a_e1"),
        answer(*ANNA_AGAIN, annotator, "a_e1"),
        answer(*BOB_FULL, annotator, "a_e2"),
        answer(*BOB_FIRST, annotator, "a_e2"),
    )


def test_missed_merge_correct_merge_and_wrong_merge():
    spans = [span(*ANNA_FULL), span(*ANNA_SHORT), span(*BOB_FULL), span(*BOB_FIRST), span(*STRANGER)]
    unlinked = {
        spans[0]: Placeholder("PERSON", 1),
        spans[1]: Placeholder("PERSON", 2),
        spans[2]: Placeholder("PERSON", 3),
        spans[3]: Placeholder("PERSON", 3),
        spans[4]: Placeholder("PERSON", 4),
    }
    counts = count_linking(DOCUMENT, gold("A1"), unlinked)["A1"]
    assert counts.unmatched_spans == 1
    assert counts.matched_spans == 4
    assert (counts.multi_mention_entities, counts.multi_mention_entities_linked) == (2, 1)
    assert (counts.multi_form_entities, counts.multi_form_entities_linked) == (2, 1)
    assert (counts.placeholders, counts.wrongly_merged_placeholders) == (3, 0)

    everything_merged = {spans[0]: Placeholder("PERSON", 1), spans[1]: Placeholder("PERSON", 1)}
    everything_merged |= {spans[2]: Placeholder("PERSON", 1), spans[3]: Placeholder("PERSON", 1), spans[4]: Placeholder("PERSON", 4)}
    counts = count_linking(DOCUMENT, gold("A1"), everything_merged)["A1"]
    assert (counts.multi_form_entities, counts.multi_form_entities_linked) == (2, 2)
    assert (counts.multi_mention_entities, counts.multi_mention_entities_linked) == (2, 2)
    assert (counts.placeholders, counts.wrongly_merged_placeholders) == (1, 1)


def test_annotators_are_compared_only_within_themselves():
    spans = [span(*ANNA_FULL), span(*BOB_FULL)]
    shared = {spans[0]: Placeholder("PERSON", 1), spans[1]: Placeholder("PERSON", 1)}
    other = (answer(*ANNA_FULL, "A2", "b_e1"), answer(*BOB_FULL, "A2", "b_e1"))
    counts = count_linking(DOCUMENT, gold("A1") + other, shared)
    assert counts["A1"].wrongly_merged_placeholders == 1
    assert counts["A2"].wrongly_merged_placeholders == 0


def test_only_direct_or_quasi_person_entities_are_gold():
    spans = [span(*ANNA_FULL)]
    answers = (answer(*ANNA_FULL, "A1", "a_e1", ReidentificationRisk.NONE),)
    assert count_linking(DOCUMENT, answers, {spans[0]: Placeholder("PERSON", 1)}) == {}


def test_report_pools_counts_and_averages_rates():
    first = LinkingCounts(multi_form_entities=1, multi_form_entities_linked=1, placeholders=2, wrongly_merged_placeholders=1)
    second = LinkingCounts(multi_form_entities=3, multi_form_entities_linked=0, placeholders=2)
    report = linking_report({"A1": first, "A2": second})
    assert report["pooled"]["multi_form_entities"] == 4
    assert report["pooled"]["linked_rate_multi_form_entities"] == 0.25
    assert report["macro_average_rates"]["linked_rate_multi_form_entities"] == 0.5
    assert report["pooled"]["wrongly_merged_rate_of_placeholders"] == 0.25
    assert report["macro_average_rates"]["linked_rate_multi_mention_entities"] is None


class StubDetector(Detector):
    """Returns the same spans for every document."""

    @property
    def name(self) -> str:
        return "stub"

    @property
    def languages(self) -> frozenset[str] | None:
        return None

    @property
    def detects(self) -> frozenset[PiiType]:
        return frozenset(PiiType)

    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans

    def load_models(self) -> None:
        pass

    def detect(self, document: Document) -> list[Span]:
        return list(self._spans)


class OneGroupResolver(IdentityResolver):
    """Puts every PERSON span into a single group whose canonical span is the longest."""

    @property
    def name(self) -> str:
        return "one-group"

    def resolve(self, document, spans):
        people = [s for s in spans if s.pii_type is PiiType.PERSON]
        return [PersonMentions(tuple(people), max(people, key=lambda s: s.end - s.start))]


def test_measure_linking_sums_documents_with_resolver_off_and_on():
    detector = StubDetector([span(*ANNA_FULL), span(*ANNA_SHORT), span(*BOB_FULL)])
    documents = [
        AnnotatedDocument(Document("doc-1", TEXT, "en"), gold("A1")),
        AnnotatedDocument(Document("doc-2", TEXT, "en"), gold("A1")),
    ]
    measurement = measure_linking(documents, detector, OneGroupResolver())

    assert measurement.off == {
        "A1": LinkingCounts(
            detected_person_spans=6,
            matched_spans=6,
            multi_mention_entities=2,
            multi_form_entities=2,
            placeholders=6,
        )
    }
    assert measurement.on == {
        "A1": LinkingCounts(
            detected_person_spans=6,
            matched_spans=6,
            multi_mention_entities=2,
            multi_mention_entities_linked=2,
            multi_form_entities=2,
            multi_form_entities_linked=2,
            placeholders=2,
            wrongly_merged_placeholders=2,
        )
    }


def test_measure_linking_counts_merged_span_when_detected_names_overlap():
    overlapping = [span(*ANNA_FULL), span(ANNA_FULL[0], ANNA_FULL[0] + 4)]
    documents = [AnnotatedDocument(DOCUMENT, gold("A1"))]
    measurement = measure_linking(documents, StubDetector(overlapping), OneGroupResolver())

    assert measurement.off["A1"].detected_person_spans == 1
    assert measurement.on["A1"].detected_person_spans == 1
