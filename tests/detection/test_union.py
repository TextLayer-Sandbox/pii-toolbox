from pii_toolbox.core import Detector, Document, PiiType, Span
from pii_toolbox.detection.union import MergingDetectorUnion, merge_overlapping


class FakeDetector(Detector):
    def __init__(self, name, spans, languages=None, detects=frozenset({PiiType.PERSON})):
        self._name = name
        self._spans = spans
        self._languages = languages
        self._detects = detects
        self.loaded = False

    @property
    def name(self):
        return self._name

    @property
    def languages(self):
        return self._languages

    @property
    def detects(self):
        return self._detects

    def load_models(self):
        self.loaded = True

    def detect(self, document):
        return list(self._spans)


DOC = Document(id="d1", text="x" * 100, language="en")


def span(start, end, pii_type, label, source, score):
    return Span(start, end, pii_type, label, (source,), score)


def test_overlap_merges_to_widest_with_longest_contributor_type():
    a = FakeDetector("a", [span(0, 12, PiiType.PERSON, "PERSON", "a", 0.9)])
    b = FakeDetector("b", [span(5, 20, PiiType.ORGANIZATION, "ORG", "b", 0.5)])
    result = MergingDetectorUnion((a, b)).detect(DOC)
    assert result == [Span(0, 20, PiiType.ORGANIZATION, "ORG", ("a", "b"), 0.9)]


def test_equal_length_overlap_takes_type_of_higher_score():
    a = FakeDetector("a", [span(0, 10, PiiType.PERSON, "PERSON", "a", 0.4)])
    b = FakeDetector("b", [span(5, 15, PiiType.LOCATION, "LOC", "b", 0.8)])
    result = MergingDetectorUnion((a, b)).detect(DOC)
    assert len(result) == 1
    assert (result[0].start, result[0].end) == (0, 15)
    assert result[0].pii_type is PiiType.LOCATION
    assert result[0].detector_label == "LOC"
    assert result[0].score == 0.8


def test_non_overlapping_spans_are_kept_sorted_by_start():
    a = FakeDetector("a", [span(30, 35, PiiType.PERSON, "PERSON", "a", 0.9)])
    b = FakeDetector("b", [span(0, 5, PiiType.LOCATION, "LOC", "b", 0.5), span(10, 12, PiiType.DATE_TIME, "DATE", "b", 0.5)])
    result = MergingDetectorUnion((a, b)).detect(DOC)
    assert [(s.start, s.end) for s in result] == [(0, 5), (10, 12), (30, 35)]


def test_touching_spans_are_not_merged():
    a = FakeDetector("a", [span(0, 5, PiiType.PERSON, "PERSON", "a", 0.9)])
    b = FakeDetector("b", [span(5, 9, PiiType.PERSON, "PERSON", "b", 0.9)])
    result = MergingDetectorUnion((a, b)).detect(DOC)
    assert [(s.start, s.end) for s in result] == [(0, 5), (5, 9)]


def test_chain_of_overlaps_merges_into_one_span():
    a = FakeDetector("a", [span(0, 10, PiiType.PERSON, "PERSON", "a", 0.3)])
    b = FakeDetector("b", [span(8, 20, PiiType.LOCATION, "LOC", "b", 0.6)])
    c = FakeDetector("c", [span(18, 40, PiiType.ORGANIZATION, "ORG", "c", 0.7)])
    result = MergingDetectorUnion((a, b, c)).detect(DOC)
    assert result == [Span(0, 40, PiiType.ORGANIZATION, "ORG", ("a", "b", "c"), 0.7)]


def test_merge_overlapping_merges_nested_and_duplicate_spans():
    outer = span(0, 5, PiiType.PERSON, "PERSON", "a", 0.5)
    inner = span(1, 2, PiiType.LOCATION, "LOC", "b", 0.9)
    assert merge_overlapping([inner, outer, outer]) == [Span(0, 5, PiiType.PERSON, "PERSON", ("a", "b"), 0.9)]


def test_members_name_languages_detects_and_load_models():
    a = FakeDetector("a", [], languages=frozenset({"en", "de"}), detects=frozenset({PiiType.PERSON}))
    b = FakeDetector("b", [], languages=frozenset({"en", "fr"}), detects=frozenset({PiiType.LOCATION}))
    union = MergingDetectorUnion((a, b))
    assert union.members == (a, b)
    assert union.name == "a+b"
    assert union.languages == frozenset({"en"})
    assert union.detects == frozenset({PiiType.PERSON, PiiType.LOCATION})
    union.load_models()
    assert a.loaded and b.loaded


def test_languages_none_only_when_all_members_support_any():
    any_language = FakeDetector("a", [])
    english = FakeDetector("b", [], languages=frozenset({"en"}))
    assert MergingDetectorUnion((any_language, any_language)).languages is None
    assert MergingDetectorUnion((any_language, english)).languages == frozenset({"en"})
