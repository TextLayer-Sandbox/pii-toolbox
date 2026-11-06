from pathlib import Path

import pytest

from pii_toolbox.core import DatasetSplit, PiiType, ReidentificationRisk
from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.datasets.tab import TabDataset

FIXTURES = Path(__file__).parent / "fixtures"
# Resolved at import, before the autouse sandbox fixture redirects the data directory.
REAL_DATA_DIR = default_data_dir()


def load_train():
    return list(TabDataset(FIXTURES).load(DatasetSplit.TRAIN))


def test_yields_three_english_documents():
    documents = load_train()
    assert [d.document.id for d in documents] == ["syn-1", "syn-2", "syn-3"]
    assert {d.document.language for d in documents} == {"en"}


def test_one_answer_per_mention_per_annotator_with_separate_boundaries():
    answers = load_train()[0].answers
    assert len(answers) == 6
    text = load_train()[0].document.text
    person = {a.annotator: text[a.start : a.end] for a in answers if a.pii_type is PiiType.PERSON}
    assert person == {"annotator1": "Anna Testwell", "annotator2": "Testwell"}


def test_labels_risk_and_entity_ids_are_kept():
    by_label = {(a.annotator, a.label): a for a in load_train()[0].answers}
    code = by_label[("annotator2", "CODE")]
    assert code.pii_type is PiiType.IDENTIFIER
    assert code.risk is ReidentificationRisk.DIRECT
    assert code.entity_id == "syn-1_a2_e2"
    quantity = by_label[("annotator1", "QUANTITY")]
    assert quantity.pii_type is PiiType.QUANTITY
    assert quantity.risk is ReidentificationRisk.NONE
    assert by_label[("annotator1", "DATETIME")].risk is ReidentificationRisk.QUASI


def test_dem_and_misc_have_no_common_type():
    answers = load_train()[1].answers
    assert {a.label: a.pii_type for a in answers if a.label in ("DEM", "MISC")} == {"DEM": None, "MISC": None}


def test_other_languages_yield_nothing():
    assert list(TabDataset(FIXTURES).load(DatasetSplit.TRAIN, "fr")) == []


@pytest.mark.data
def test_real_corpus_split_sizes():
    dataset = TabDataset(REAL_DATA_DIR)
    sizes = [sum(1 for _ in dataset.load(split)) for split in DatasetSplit]
    assert sizes == [1014, 127, 127]


def test_annotates_every_common_type():
    assert TabDataset(FIXTURES).annotated_types == frozenset(PiiType)
