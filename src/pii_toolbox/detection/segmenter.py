"""Splits long text into overlapping segments and runs a detector over each one.

Models read a limited input length, so long documents are detected piece by piece and the
spans are mapped back to positions in the full text.
"""

from collections.abc import Callable
from dataclasses import replace

from pii_toolbox.core import Span, TextSegment, TextSegmenter


class OverlappingTextSegmenter(TextSegmenter):
    """Cuts text into segments of at most `length` characters sharing `overlap` characters.

    Each cut falls on whitespace when the last half of the segment holds any, so words
    are rarely split; the overlap lets an item cut by one segment edge appear whole in the
    next segment.
    """

    def __init__(self, length: int = 2000, overlap: int = 200) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        if not 0 <= overlap < length:
            raise ValueError("overlap must be at least 0 and less than length")
        self._length = length
        self._overlap = overlap

    def split(self, text: str) -> list[TextSegment]:
        segments: list[TextSegment] = []
        start = 0
        while True:
            end = min(start + self._length, len(text))
            if end < len(text):
                end = self._whitespace_cut(text, start, end)
            segments.append(TextSegment(offset=start, text=text[start:end]))
            if end >= len(text):
                return segments
            start = max(end - self._overlap, start + 1)

    def _whitespace_cut(self, text: str, start: int, end: int) -> int:
        earliest = start + self._length // 2
        for position in range(end, earliest, -1):
            if text[position - 1].isspace():
                return position
        return end


def detect_segmented(
    text: str,
    segmenter: TextSegmenter,
    detect_segment: Callable[[str], list[Span]],
) -> list[Span]:
    """Detects in every segment and returns spans at full-text positions, sorted by start.

    A span found twice in an overlap (same start, end and type) is kept once. A span cut by
    a segment edge is dropped when another span found in a later or earlier segment contains it.
    """
    found: dict[tuple[int, int, object], tuple[Span, bool]] = {}
    segments = segmenter.split(text)
    for segment in segments:
        segment_end = segment.offset + len(segment.text)
        for span in detect_segment(segment.text):
            shifted = replace(span, start=span.start + segment.offset, end=span.end + segment.offset)
            touches_inner_edge = (span.start == 0 and segment.offset > 0) or (
                shifted.end == segment_end and segment_end < len(text)
            )
            key = (shifted.start, shifted.end, shifted.pii_type)
            previous = found.get(key)
            if previous is None:
                found[key] = (shifted, touches_inner_edge)
            else:
                found[key] = (previous[0], previous[1] and touches_inner_edge)

    spans = [span for span, _ in found.values()]
    kept = [
        span
        for span, cut in found.values()
        if not cut
        or not any(
            other is not span and other.start <= span.start and span.end <= other.end for other in spans
        )
    ]
    return sorted(kept, key=lambda span: (span.start, span.end))
