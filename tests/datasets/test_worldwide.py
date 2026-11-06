import io
import tarfile

import pytest

from pii_toolbox.core import DatasetSplit, PiiType
from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.datasets.worldwide import WorldwideDataset

# Resolved at import, before the autouse sandbox fixture redirects the data directory.
REAL_DATA_DIR = default_data_dir()

ROOT = "en-worldwide-newswire-0123abc"


def write_tarball(data_dir, articles):
    folder = data_dir / "worldwide"
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(folder / "en-worldwide-newswire.tar.gz", "w:gz") as archive:
        for name, content in articles.items():
            payload = content.encode("utf-8")
            member = tarfile.TarInfo(f"{ROOT}/processed_annotated/{name}.txt.tsv")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        readme = tarfile.TarInfo(f"{ROOT}/README.md")
        readme.size = 2
        archive.addfile(readme, io.BytesIO(b"hi"))


def tsv(*lines):
    return "\n".join(lines) + "\n"


def load_all(data_dir, split=DatasetSplit.TEST, language=None):
    return list(WorldwideDataset(data_dir).load(split, language))


def spans(annotated):
    text = annotated.document.text
    return [(text[a.start : a.end], a.pii_type, a.label) for a in annotated.answers]


def test_one_document_per_article_in_name_order(tmp_path):
    write_tarball(tmp_path, {"zz_second": tsv("Hello\tO"), "aa_first": tsv("Hi\tO")})
    documents = load_all(tmp_path)
    assert [d.document.id for d in documents] == ["aa_first", "zz_second"]
    assert all(d.document.language == "en" for d in documents)


def test_split_is_ignored(tmp_path):
    write_tarball(tmp_path, {"a": tsv("Hi\tO")})
    assert {len(load_all(tmp_path, split)) for split in DatasetSplit} == {1}


def test_person_span_text_and_type(tmp_path):
    write_tarball(tmp_path, {"a": tsv("Anna\tB-Person", "Smith\tI-Person", "left\tO")})
    (annotated,) = load_all(tmp_path)
    assert spans(annotated) == [("Anna Smith", PiiType.PERSON, "Person")]


def test_date_and_money_are_mapped(tmp_path):
    write_tarball(tmp_path, {"a": tsv("on\tO", "May\tB-Date", "paid\tO", "5\tB-Money", "dollars\tI-Money")})
    (annotated,) = load_all(tmp_path)
    assert spans(annotated) == [
        ("May", PiiType.DATE_TIME, "Date"),
        ("5 dollars", PiiType.QUANTITY, "Money"),
    ]


def test_facility_has_no_common_type(tmp_path):
    write_tarball(tmp_path, {"a": tsv("Harbor\tB-Facility", "Bridge\tI-Facility")})
    (annotated,) = load_all(tmp_path)
    assert spans(annotated) == [("Harbor Bridge", None, "Facility")]


def test_extra_columns_are_ignored(tmp_path):
    write_tarball(tmp_path, {"a": tsv("Anna\tB-Person\tB-Misc", "Smith\tI-Person\tI-Misc\tO")})
    (annotated,) = load_all(tmp_path)
    assert spans(annotated) == [("Anna Smith", PiiType.PERSON, "Person")]


def test_lines_with_fewer_than_two_columns_are_skipped(tmp_path):
    write_tarball(tmp_path, {"a": tsv("Anna\tB-Person", "stray", "Smith\tI-Person")})
    (annotated,) = load_all(tmp_path)
    assert annotated.document.text == "Anna Smith"


def test_blank_line_gives_a_newline_and_offsets_stay_right(tmp_path):
    write_tarball(tmp_path, {"a": tsv("Hello\tO", "there\tO", "", "Anna\tB-Person", "left\tO")})
    (annotated,) = load_all(tmp_path)
    assert annotated.document.text == "Hello there\nAnna left"
    assert spans(annotated) == [("Anna", PiiType.PERSON, "Person")]


def test_other_languages_yield_nothing(tmp_path):
    write_tarball(tmp_path, {"a": tsv("Hi\tO")})
    assert load_all(tmp_path, language="fr") == []
    assert len(load_all(tmp_path, language="en")) == 1


def test_name_and_languages():
    dataset = WorldwideDataset(None)
    assert dataset.name.lower() == "worldwide"
    assert dataset.languages == frozenset({"en"})


@pytest.mark.data
def test_real_download_has_every_article():
    assert len(load_all(REAL_DATA_DIR)) == 1069


def test_annotates_every_common_type_except_identifier():
    assert WorldwideDataset(REAL_DATA_DIR).annotated_types == frozenset(PiiType) - {PiiType.IDENTIFIER}
