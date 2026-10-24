import json

import pytest

from pii_toolbox.core import DatasetSplit, PiiType
from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.datasets.mapa import MapaDataset, spans_from_bio

# Resolved at import, before the autouse sandbox fixture redirects the data directory.
REAL_DATA_DIR = default_data_dir()

TOKENS = ["Anna", "Smith", "visited", "Paris", "on", "1", "May"]
TAGS = ["B-PERSON", "I-PERSON", "O", "B-ADDRESS", "O", "B-DATE", "I-DATE"]


def sentence(language, file_name, number, tokens, tags):
    return {
        "language": language,
        "type": "EUR-LEX",
        "file_name": file_name,
        "sentence_number": number,
        "tokens": tokens,
        "coarse_grained": tags,
        "fine_grained": ["O"] * len(tokens),
    }


def write_corpus(tmp_path, split_file, records):
    folder = tmp_path / "mapa"
    folder.mkdir(exist_ok=True)
    (folder / split_file).write_text("\n".join(json.dumps(r) for r in records))


def test_spans_are_rebuilt_at_character_offsets():
    text = " ".join(TOKENS)
    spans = [(text[s:e], label) for s, e, label in spans_from_bio(TOKENS, TAGS)]
    assert spans == [("Anna Smith", "PERSON"), ("Paris", "ADDRESS"), ("1 May", "DATE")]


def test_dangling_inside_tag_starts_a_span():
    assert spans_from_bio(["Anna", "Smith"], ["I-PERSON", "I-PERSON"]) == [(0, 10, "PERSON")]


def test_inside_tag_of_another_label_starts_a_new_span():
    assert spans_from_bio(["Paris", "May"], ["B-ADDRESS", "I-DATE"]) == [(0, 5, "ADDRESS"), (6, 9, "DATE")]


def test_loaded_answers_are_mapped_to_common_types(tmp_path):
    write_corpus(tmp_path, "validation.jsonl", [sentence("en", "f1", 0, TOKENS, TAGS)])
    document = next(MapaDataset(tmp_path).load(DatasetSplit.DEV))
    text = document.document.text
    found = [(text[a.start : a.end], a.pii_type) for a in document.answers]
    assert found == [
        ("Anna Smith", PiiType.PERSON),
        ("Paris", PiiType.LOCATION),
        ("1 May", PiiType.DATE_TIME),
    ]


def test_time_and_date_both_map_to_date_time(tmp_path):
    write_corpus(
        tmp_path, "validation.jsonl", [sentence("en", "f1", 0, ["at", "9:00", "on", "May"], ["O", "B-TIME", "O", "B-DATE"])]
    )
    answers = next(MapaDataset(tmp_path).load(DatasetSplit.DEV)).answers
    assert [a.pii_type for a in answers] == [PiiType.DATE_TIME, PiiType.DATE_TIME]


def test_sentences_of_a_file_form_one_document_with_global_offsets(tmp_path):
    records = [
        sentence("en", "f1", 1, ["Paris", "again"], ["B-ADDRESS", "O"]),
        sentence("en", "f1", 0, ["Hello", "world"], ["O", "O"]),
    ]
    write_corpus(tmp_path, "validation.jsonl", records)
    document = next(MapaDataset(tmp_path).load(DatasetSplit.DEV))
    assert document.document.text == "Hello world\nParis again"
    answer = document.answers[0]
    assert document.document.text[answer.start : answer.end] == "Paris"


def test_language_filter_and_dev_is_validation_file(tmp_path):
    write_corpus(
        tmp_path,
        "validation.jsonl",
        [sentence("fr", "f1", 0, ["Paris"], ["B-ADDRESS"]), sentence("de", "f2", 0, ["Berlin"], ["B-ADDRESS"])],
    )
    write_corpus(tmp_path, "train.jsonl", [sentence("de", "f3", 0, ["Bonn"], ["B-ADDRESS"])])
    dataset = MapaDataset(tmp_path)
    assert [d.document.language for d in dataset.load(DatasetSplit.DEV, "fr")] == ["fr"]
    assert [d.document.id for d in dataset.load(DatasetSplit.DEV)] == ["f1", "f2"]
    assert [d.document.id for d in dataset.load(DatasetSplit.TRAIN)] == ["f3"]


@pytest.mark.data
def test_real_corpus_has_each_headline_language_per_split():
    dataset = MapaDataset(REAL_DATA_DIR)
    for split in DatasetSplit:
        for language in ("en", "fr", "de"):
            assert next(dataset.load(split, language), None) is not None
