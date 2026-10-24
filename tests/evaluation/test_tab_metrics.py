import pytest

from pii_toolbox.core import AnnotatedDocument, AnswerSpan, Document, PiiType, ReidentificationRisk, Span
from pii_toolbox.evaluation.tab_metrics import OfficialTabMetricsScorer, TokenInfo

DIRECT = ReidentificationRisk.DIRECT
QUASI = ReidentificationRisk.QUASI
NO_MASK = ReidentificationRisk.NONE

# "Anna Testwell met the Invented Ministry in Exampleton. Anna Testwell left."
TEXT = "Anna Testwell met the Invented Ministry in Exampleton. Anna Testwell left."


def stub_tokens(text):
    """Tags only 'the' (DET) and 'in' (ADP) as ignorable function words."""
    tokens = []
    position = 0
    for word in text.split(" "):
        tag = "DET" if word == "the" else "ADP" if word == "in" else "X"
        tokens.append(TokenInfo(position, position + len(word), tag, word.lower()))
        position += len(word) + 1
    return tokens


def span(start, end):
    return Span(start, end, PiiType.PERSON, "PERSON", ("test",), None)


def mention(start, end, risk, entity_id, annotator="A"):
    return AnswerSpan(start, end, PiiType.PERSON, "PERSON", annotator, risk, entity_id)


def at(text):
    return TEXT.index(text)


def annotated(answers):
    return AnnotatedDocument(Document("d", TEXT, "en"), tuple(answers))


def score(answers, spans, **kwargs):
    scorer = OfficialTabMetricsScorer(stub_tokens, **kwargs)
    return scorer.score([(annotated(answers), spans)])


def person_mentions(risk=DIRECT, annotator="A", prefix="e1"):
    first = at("Anna Testwell")
    second = TEXT.rindex("Anna Testwell")
    return [mention(first, first + 13, risk, prefix, annotator), mention(second, second + 13, risk, prefix, annotator)]


def test_entity_is_found_only_if_every_mention_is_masked():
    first, second = at("Anna Testwell"), TEXT.rindex("Anna Testwell")
    assert score(person_mentions(), [span(first, first + 13), span(second, second + 13)]).er_di == 1.0
    assert score(person_mentions(), [span(first, first + 13)]).er_di == 0.0


def test_direct_and_quasi_entities_are_scored_separately():
    town = at("Exampleton")
    answers = person_mentions(DIRECT) + [mention(town, town + 10, QUASI, "e2")]
    metrics = score(answers, [span(town, town + 10)])
    assert metrics.er_di == 0.0
    assert metrics.er_qi == 1.0


def test_function_words_and_punctuation_inside_a_mention_may_stay_unmasked():
    start = at("the Invented Ministry")
    answers = [mention(start, start + 21, QUASI, "e1")]
    inner = start + 4
    assert score(answers, [span(inner, start + 21)]).er_qi == 1.0


def test_content_word_left_unmasked_is_a_miss():
    start = at("the Invented Ministry")
    answers = [mention(start, start + 21, QUASI, "e1")]
    assert score(answers, [span(start + 4, start + 12)]).er_qi == 0.0


def test_no_mask_entities_are_not_counted():
    start = at("Exampleton")
    metrics = score([mention(start, start + 10, NO_MASK, "e1")], [])
    assert (metrics.er_di, metrics.er_qi) == (0.0, 0.0)


def test_mention_not_needing_masking_may_stay_open():
    first, second = at("Anna Testwell"), TEXT.rindex("Anna Testwell")
    answers = [mention(first, first + 13, DIRECT, "e1"), mention(second, second + 13, NO_MASK, "e1")]
    assert score(answers, [span(first, first + 13)]).er_di == 1.0


def test_recall_pools_entities_over_annotators():
    first, second = at("Anna Testwell"), TEXT.rindex("Anna Testwell")
    answers = person_mentions(DIRECT, "A", "a1") + person_mentions(DIRECT, "B", "b1")[:1]
    both = [span(first, first + 13), span(second, second + 13)]
    # Annotator A's entity is found; annotator B's single mention is also covered.
    assert score(answers, both).er_di == 1.0
    assert score(answers, [span(first, first + 13)]).er_di == 0.5


def test_precision_counts_tokens_by_share_of_annotators_covering_them():
    first = at("Anna Testwell")
    answers = [
        mention(first, first + 13, DIRECT, "a1", "A"),
        mention(first, first + 4, DIRECT, "b1", "B"),
    ]
    # Tokens Anna (A and B cover) and Testwell (A only): (2 + 1) / (2 + 2).
    assert score(answers, [span(first, first + 13)]).wp == pytest.approx(3 / 4)


def test_precision_uses_the_token_weight_function():
    first = at("Anna Testwell")
    answers = [mention(first, first + 4, DIRECT, "a1", "A")]

    def weigh(text, spans):
        return [3.0 if text[s:e] == "Anna" else 1.0 for s, e in spans]

    metrics = score(answers, [span(first, first + 13)], weigh_tokens=weigh)
    assert metrics.wp == pytest.approx(3 / 4)


def test_empty_input_scores_zero():
    metrics = OfficialTabMetricsScorer(stub_tokens).score([])
    assert (metrics.er_di, metrics.er_qi, metrics.wp) == (0.0, 0.0, 0.0)
