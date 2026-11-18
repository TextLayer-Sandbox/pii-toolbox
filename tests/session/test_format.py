from pii_toolbox.core import Placeholder
from pii_toolbox.session import AngleBracketFormat

FORMAT = AngleBracketFormat()


def test_render_uses_angle_brackets():
    assert FORMAT.render(Placeholder("PERSON", 1)) == "<PERSON_1>"


def test_parse_tolerates_dropped_brackets_and_case():
    text = "hi <PERSON_1>, PERSON_1 and person_1"
    tokens = FORMAT.parse(text)
    assert [text[t.start : t.end] for t in tokens] == ["<PERSON_1>", "PERSON_1", "person_1"]
    assert all(t.placeholder == Placeholder("PERSON", 1) for t in tokens)


def test_parse_takes_all_digits_of_the_number():
    tokens = FORMAT.parse("PERSON_10")
    assert [t.placeholder for t in tokens] == [Placeholder("PERSON", 10)]


def test_parse_ignores_labels_that_are_not_pii_types():
    assert FORMAT.parse("TEAM_7 <EMAIL_ADDRESS_2>") == []


def test_parse_label_with_underscore():
    tokens = FORMAT.parse("<IDENTIFIER_2>")
    assert [t.placeholder for t in tokens] == [Placeholder("IDENTIFIER", 2)]
    assert (tokens[0].start, tokens[0].end) == (0, 14)


def test_parse_ignores_token_glued_to_a_longer_word():
    tokens = FORMAT.parse("XPERSON_1")
    assert Placeholder("PERSON", 1) not in [t.placeholder for t in tokens]


def test_parse_ignores_digits_glued_after_number():
    assert FORMAT.parse("PERSON_1x") == []
