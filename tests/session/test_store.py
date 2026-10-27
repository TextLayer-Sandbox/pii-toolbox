from pii_toolbox.core import Placeholder
from pii_toolbox.session import InMemoryPlaceholderStore


def test_first_mint_then_reuse():
    store = InMemoryPlaceholderStore()
    assert store.get_or_mint("s1", "PERSON", "David") == (Placeholder("PERSON", 1), True)
    assert store.get_or_mint("s1", "PERSON", "David") == (Placeholder("PERSON", 1), False)


def test_numbering_is_per_label():
    store = InMemoryPlaceholderStore()
    store.get_or_mint("s1", "PERSON", "David")
    assert store.get_or_mint("s1", "PERSON", "Maria")[0] == Placeholder("PERSON", 2)
    assert store.get_or_mint("s1", "EMAIL_ADDRESS", "a@example.test")[0] == Placeholder("EMAIL_ADDRESS", 1)


def test_sessions_number_independently():
    store = InMemoryPlaceholderStore()
    store.get_or_mint("s1", "PERSON", "David")
    store.get_or_mint("s1", "PERSON", "Maria")
    assert store.get_or_mint("s2", "PERSON", "Maria")[0] == Placeholder("PERSON", 1)


def test_alias_shares_placeholder_and_original_is_fullest_form():
    store = InMemoryPlaceholderStore()
    p1, _ = store.get_or_mint("s1", "PERSON", "David")
    store.alias("s1", "Dave", p1)
    assert store.get_or_mint("s1", "PERSON", "Dave") == (p1, False)
    assert store.original("s1", p1) == "David"
    store.alias("s1", "David Parker", p1)
    assert store.original("s1", p1) == "David Parker"


def test_labels_lists_minted_labels():
    store = InMemoryPlaceholderStore()
    assert store.labels("s1") == frozenset()
    store.get_or_mint("s1", "PERSON", "David")
    store.get_or_mint("s1", "IBAN_CODE", "XX00 0000")
    assert store.labels("s1") == frozenset({"PERSON", "IBAN_CODE"})


def test_discard_empties_one_session_only():
    store = InMemoryPlaceholderStore()
    p1, _ = store.get_or_mint("s1", "PERSON", "David")
    p2, _ = store.get_or_mint("s2", "PERSON", "Maria")
    store.discard("s1")
    assert store.original("s1", p1) is None
    assert store.labels("s1") == frozenset()
    assert store.get_or_mint("s1", "PERSON", "Maria")[0] == Placeholder("PERSON", 1)
    assert store.original("s2", p2) == "Maria"


def test_repr_hides_originals():
    store = InMemoryPlaceholderStore()
    store.get_or_mint("s1", "PERSON", "Zanzibar Quill")
    assert "Zanzibar" not in repr(store)
    assert "Zanzibar" not in repr(store._sessions["s1"])


def test_no_file_is_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = InMemoryPlaceholderStore()
    store.get_or_mint("s1", "PERSON", "David")
    store.discard("s1")
    assert list(tmp_path.iterdir()) == []
