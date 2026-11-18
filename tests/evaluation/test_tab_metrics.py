import importlib.util
import itertools

import pytest

from pii_toolbox.core import AnnotatedDocument, AnswerSpan, DatasetSplit, Document, PiiType, ReidentificationRisk, Span
from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.datasets.tab import TabDataset
from pii_toolbox.core.model_dir import default_model_dir
from pii_toolbox.evaluation.tab_metrics import (
    BertInformationWeigher,
    OfficialTabMetricsScorer,
    TokenInfo,
    spacy_token_annotator,
)

REAL_DATA_DIR = default_data_dir()
REAL_MODEL_DIR = default_model_dir()

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


def _load_tab_script(data_dir):
    spec = importlib.util.spec_from_file_location("tab_evaluation_script", data_dir / "tab" / "evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_predictions(documents):
    """Masks every other masking mention of the first annotator, trims the first word off every fourth, and adds one unrelated word run."""
    predictions = {}
    for annotated in documents:
        first_annotator = annotated.answers[0].annotator
        mentions = [
            a for a in annotated.answers if a.annotator == first_annotator and a.risk in (DIRECT, QUASI)
        ]
        spans = [(a.start, a.end) for a in mentions[::2]]
        for mention in mentions[1::4]:
            text = annotated.document.text[mention.start : mention.end]
            first_space = text.find(" ")
            if first_space != -1:
                spans.append((mention.start + first_space + 1, mention.end))
        spans.append((0, min(30, len(annotated.document.text))))
        predictions[annotated.document.id] = spans
    return predictions


@pytest.mark.data
@pytest.mark.model
def test_scores_equal_tab_own_script_on_a_published_sample(monkeypatch):
    data_dir = REAL_DATA_DIR
    model_dir = REAL_MODEL_DIR
    script = _load_tab_script(data_dir)
    documents = list(itertools.islice(TabDataset(data_dir).load(DatasetSplit.TEST), 3))
    predictions = _sample_predictions(documents)

    unpacked = model_dir / "en_core_web_md" / "unpacked"
    annotate = spacy_token_annotator(model_dir)  # unpacks the wheel on first use
    spacy_path = next(unpacked.glob("en_core_web_md/en_core_web_md-*"))
    corpus = script.GoldCorpus(str(data_dir / "tab" / "echr_test.json"), spacy_model=str(spacy_path))
    masked = [script.MaskedDocument(doc_id, spans) for doc_id, spans in predictions.items()]

    weigher = BertInformationWeigher(model_dir)
    tab_weighting = object.__new__(script.BertTokenWeighting)
    tab_weighting.tokeniser = weigher._tokenizer
    tab_weighting.model = weigher._model
    tab_weighting.device = "cpu"
    tab_weighting.max_segment_size = 100

    expected_er_di = corpus.get_entity_recall(masked, True, False)
    expected_er_qi = corpus.get_entity_recall(masked, False, True)
    expected_wp = corpus.get_precision(masked, tab_weighting)

    results = [
        (annotated, [Span(start, end, PiiType.PERSON, "PERSON", ("test",), None) for start, end in predictions[annotated.document.id]])
        for annotated in documents
    ]
    metrics = OfficialTabMetricsScorer(annotate, weigher).score(results)

    assert metrics.er_di == pytest.approx(expected_er_di, abs=1e-4)
    assert metrics.er_qi == pytest.approx(expected_er_qi, abs=1e-4)
    assert metrics.wp == pytest.approx(expected_wp, abs=1e-4)
