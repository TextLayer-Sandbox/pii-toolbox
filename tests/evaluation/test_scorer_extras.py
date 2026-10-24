import pytest

from pii_toolbox.core import AnswerSpan, Document, PiiType, ReidentificationRisk, Span, SpanMatchRule
from pii_toolbox.evaluation.report_extras import score_with_extras

DIRECT = ReidentificationRisk.DIRECT


def predicted(start, end, pii_type=PiiType.PERSON):
    return Span(start, end, pii_type, pii_type.value, ("test",), None)


def answer(start, end, pii_type, label=None):
    return AnswerSpan(start, end, pii_type, label or (pii_type.value if pii_type else "DEM"), "A", DIRECT, None)


def document(spans, answers, language="en"):
    return (Document("d", "x" * 500, language), spans, answers)


def shared_only():
    answers = [answer(0, 5, PiiType.PERSON), answer(10, 15, PiiType.LOCATION), answer(20, 25, PiiType.PERSON)]
    spans = [predicted(0, 5), predicted(10, 15, PiiType.LOCATION), predicted(300, 305)]
    return document(spans, answers)


def with_extras():
    base = shared_only()
    extra_answers = [answer(40, 45, None, "DEM"), answer(50, 55, None, "DEM"), answer(60, 65, None, "MISC")]
    extra_spans = [predicted(40, 45, PiiType.QUANTITY)]
    return document(base[1] + extra_spans, base[2] + extra_answers)


def score(results, dataset="tab", rule=SpanMatchRule.ANY_OVERLAP):
    return score_with_extras(results, rule, dataset=dataset)


def test_dem_and_misc_are_extra_rows_with_their_own_scores():
    report = score([with_extras()])
    assert set(report.extra_rows) == {"DEM", "MISC"}
    dem = report.extra_rows["DEM"]
    assert dem.recall == 0.5
    assert report.extra_rows["MISC"].recall == 0.0


def test_extra_row_precision_counts_predictions_overlapping_nothing():
    report = score([with_extras()])
    # One DEM hit; the unexplained prediction at 300..305 is the only false positive.
    assert report.extra_rows["DEM"].precision == pytest.approx(1 / 2)


def test_shared_overall_is_unchanged_by_extra_answers():
    base = shared_only()
    extra_answers = [answer(40, 45, None, "DEM"), answer(60, 65, None, "MISC")]
    plain = score([base])
    extended = score([document(base[1], base[2] + extra_answers)])
    assert extended.shared_overall_micro == plain.shared_overall_micro
    assert extended.shared_macro == plain.shared_macro


def test_overall_with_identifier_exists_for_tab_only():
    answers = [answer(0, 5, PiiType.PERSON), answer(10, 18, PiiType.IDENTIFIER)]
    results = [document([predicted(0, 5)], answers)]
    tab = score(results, "tab")
    mapa = score(results, "mapa")
    assert tab.overall_with_identifier_micro.recall == 0.5
    assert tab.shared_overall_micro.recall == 1.0
    assert mapa.overall_with_identifier_micro is None
    assert mapa.overall_with_identifier_macro is None


def test_identifier_never_appears_in_mapa_results():
    results = [document([predicted(10, 18, PiiType.IDENTIFIER)], [answer(0, 5, PiiType.PERSON)])]
    mapa = score(results, "mapa")
    assert PiiType.IDENTIFIER not in mapa.by_type
    assert mapa.shared_overall_micro.precision == 0.0
    assert mapa.extra_rows == {}
