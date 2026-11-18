from pii_toolbox.core import Placeholder
from pii_toolbox.session import AngleBracketFormat

FORMAT = AngleBracketFormat()


def test_render_uses_angle_brackets():
    assert FORMAT.render(Placeholder("PERSON", 1)) == "<PERSON_1>"


def test_parse_accepts_only_the_exact_rendered_form():
    text = "<PERSON_1> PERSON_1 person_1 Person_1 <person_1> <PERSON_01> <PERSON_1 PERSON_1>"
    tokens = FORMAT.parse(text)
    assert [text[t.start : t.end] for t in tokens] == ["<PERSON_1>"]
    assert tokens[0].placeholder == Placeholder("PERSON", 1)


def test_parse_takes_all_digits_of_the_number():
    tokens = FORMAT.parse("<PERSON_10>")
    assert [t.placeholder for t in tokens] == [Placeholder("PERSON", 10)]


def test_parse_ignores_labels_that_are_not_pii_types():
    assert FORMAT.parse("TEAM_7 <EMAIL_ADDRESS_2>") == []


def test_parse_label_with_underscore():
    tokens = FORMAT.parse("<IDENTIFIER_2>")
    assert [t.placeholder for t in tokens] == [Placeholder("IDENTIFIER", 2)]
    assert (tokens[0].start, tokens[0].end) == (0, 14)


def test_parse_finds_token_glued_to_surrounding_letters():
    text = "x<PERSON_1>s"
    tokens = FORMAT.parse(text)
    assert [text[t.start : t.end] for t in tokens] == ["<PERSON_1>"]
