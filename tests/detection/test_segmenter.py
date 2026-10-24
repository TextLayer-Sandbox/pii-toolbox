import re

from pii_toolbox.core import PiiType, Span
from pii_toolbox.detection.segmenter import OverlappingTextSegmenter, detect_segmented

MARKER = "ZX-1234-QQ"


def marker_detector(segment_text):
    return [
        Span(m.start(), m.end(), PiiType.IDENTIFIER, "MARKER", ("fake",), 1.0)
        for m in re.finditer(re.escape(MARKER), segment_text)
    ]


def build_text(total, every):
    pieces = []
    length = 0
    next_marker = every
    index = 0
    while length < total:
        word = f"w{index % 97} "
        if length + len(word) >= next_marker:
            pieces.append(MARKER)
            length += len(MARKER)
            next_marker += every
        else:
            pieces.append(word)
            length += len(word)
        index += 1
    return "".join(pieces)


def test_long_text_yields_one_span_per_marker_at_full_text_offsets():
    text = build_text(100_000, 997)
    segmenter = OverlappingTextSegmenter(length=2000, overlap=200)
    segments = segmenter.split(text)
    assert segments[0].offset == 0
    assert segments[-1].offset + len(segments[-1].text) == len(text)
    for earlier, later in zip(segments, segments[1:]):
        assert later.offset <= earlier.offset + len(earlier.text)
    expected = [m.start() for m in re.finditer(re.escape(MARKER), text)]
    assert len(expected) > 90
    spans = detect_segmented(text, segmenter, marker_detector)
    assert [s.start for s in spans] == expected
    assert all(text[s.start : s.end] == MARKER for s in spans)


def test_text_shorter_than_one_segment_is_a_single_segment():
    text = f"hello {MARKER} bye"
    segmenter = OverlappingTextSegmenter(length=2000, overlap=200)
    assert len(segmenter.split(text)) == 1
    spans = detect_segmented(text, segmenter, marker_detector)
    assert [(s.start, s.end) for s in spans] == [(6, 6 + len(MARKER))]


def test_marker_exactly_across_a_join_is_found_whole():
    text = "a" * 16 + MARKER + "b" * 30
    segmenter = OverlappingTextSegmenter(length=20, overlap=12)
    segments = segmenter.split(text)
    assert not any(MARKER in segment.text and segment.offset == 0 for segment in segments)
    spans = detect_segmented(text, segmenter, marker_detector)
    assert [(s.start, s.end) for s in spans] == [(16, 16 + len(MARKER))]


def test_span_cut_by_a_segment_edge_is_dropped_when_seen_whole_elsewhere():
    text = "a" * 16 + MARKER + "b" * 30

    def detector_with_partials(segment_text):
        spans = marker_detector(segment_text)
        cut = segment_text.find(MARKER[:4])
        if cut != -1 and MARKER not in segment_text:
            spans.append(Span(cut, len(segment_text), PiiType.IDENTIFIER, "MARKER", ("fake",), 1.0))
        return spans

    spans = detect_segmented(text, OverlappingTextSegmenter(length=20, overlap=12), detector_with_partials)
    assert [(s.start, s.end) for s in spans] == [(16, 16 + len(MARKER))]
