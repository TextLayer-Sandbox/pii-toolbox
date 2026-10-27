import dataclasses

import pytest

from pii_toolbox.core import PiiType, Span


def make_span(start: int, end: int) -> Span:
    return Span(start, end, PiiType.PERSON, "PERSON", ("test",), None)


def test_span_is_frozen():
    span = make_span(0, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.start = 1  # type: ignore[misc]


@pytest.mark.parametrize("start, end", [(5, 5), (-1, 3), (6, 5)])
def test_span_rejects_invalid_bounds(start, end):
    with pytest.raises(ValueError):
        make_span(start, end)
