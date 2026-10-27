"""Tests for the default nickname list and the loader that validates a list file's shape."""

from pathlib import Path

import pytest

from pii_toolbox.coref.nicknames import DEFAULT_NICKNAME_LIST_PATH, load_nickname_list


@pytest.fixture(scope="module")
def default_list() -> dict[str, str]:
    return load_nickname_list(DEFAULT_NICKNAME_LIST_PATH)


def write_list(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "list.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_matt_and_matty_map_to_matthew(default_list):
    assert default_list["matt"] == "matthew"
    assert default_list["matty"] == "matthew"


def test_dave_maps_to_david(default_list):
    assert default_list["dave"] == "david"


def test_no_name_is_both_standard_and_nickname(default_list):
    assert set(default_list.values()) & set(default_list) == set()


def test_default_file_has_one_row_per_nickname():
    rows = [line.split(",") for line in DEFAULT_NICKNAME_LIST_PATH.read_text(encoding="utf-8").splitlines()]
    nicknames = [row[0] for row in rows]
    assert all(len(row) == 2 for row in rows)
    assert len(nicknames) == len(set(nicknames))


def test_licence_notice_sits_beside_the_data_file():
    notice = DEFAULT_NICKNAME_LIST_PATH.with_name("NICKNAMES_LICENSE").read_text(encoding="utf-8")
    assert "carltonnorthern/nicknames" in notice
    assert "Apache License" in notice


def test_loader_reads_a_valid_file(tmp_path):
    path = write_list(tmp_path, "Matt,Matthew\nmatty,matthew\n\ndave,david\n")
    assert load_nickname_list(path) == {"matt": "matthew", "matty": "matthew", "dave": "david"}


def test_loader_rejects_a_nickname_with_two_standard_names(tmp_path):
    path = write_list(tmp_path, "matt,matthew\nmatt,matthias\n")
    with pytest.raises(ValueError, match="matt"):
        load_nickname_list(path)


def test_loader_rejects_a_name_that_is_standard_and_nickname(tmp_path):
    path = write_list(tmp_path, "bob,robert\nrobert,bob\n")
    with pytest.raises(ValueError, match="both a standard name and a nickname"):
        load_nickname_list(path)


def test_loader_rejects_a_chain(tmp_path):
    path = write_list(tmp_path, "bobby,bob\nbob,robert\n")
    with pytest.raises(ValueError, match="bob"):
        load_nickname_list(path)


def test_loader_rejects_a_malformed_row(tmp_path):
    path = write_list(tmp_path, "matt,matthew,extra\n")
    with pytest.raises(ValueError, match="expected"):
        load_nickname_list(path)
