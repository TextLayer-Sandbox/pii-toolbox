import json

import pytest

from pii_toolbox.datasets.structured_pii import (
    GRETEL_REVISION,
    NEMOTRON_REVISION,
    STRUCTURED_DATASET_FILES,
    LabelledSpan,
    download_structured_datasets,
    load_gretel_test,
    load_nemotron_test,
    parse_gretel_spans,
    parse_nemotron_spans,
)


def test_nemotron_spans_accept_numeric_text_and_drop_unreadable_ones():
    raw = repr([
        {"start": 0, "end": 5, "label": "email", "text": "x"},
        {"start": "7", "end": "9", "label": "url", "text": 3},
        {"start": "a", "end": 4, "label": "url"},
        {"start": 5, "end": 5, "label": "url"},
    ])
    assert parse_nemotron_spans(raw) == (LabelledSpan(0, 5, "email"), LabelledSpan(7, 9, "url"))


def test_malformed_spans_give_no_spans():
    assert parse_nemotron_spans("not a list [") == ()
    assert parse_gretel_spans("{bad") == ()


def test_gretel_spans_are_json_with_exclusive_end():
    raw = json.dumps([{"start": 3, "end": 8, "label": "iban"}])
    assert parse_gretel_spans(raw) == (LabelledSpan(3, 8, "iban"),)


def test_downloads_are_pinned_to_revisions(tmp_path):
    urls = [item.url for item in STRUCTURED_DATASET_FILES]
    assert any(NEMOTRON_REVISION in url for url in urls)
    assert any(GRETEL_REVISION in url for url in urls)


def write_parquet(path, columns):
    import pyarrow
    import pyarrow.parquet as parquet

    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(pyarrow.table(columns), path)


def test_loaders_read_the_parquet_files(tmp_path):
    write_parquet(
        tmp_path / "structured" / "nemotron" / "test.parquet",
        {"text": ["a@b.co"], "spans": [repr([{"start": 0, "end": 6, "label": "email"}])], "locale": ["us"]},
    )
    write_parquet(
        tmp_path / "structured" / "gretel" / "test.parquet",
        {
            "generated_text": ["DE89 3704 0044 0532 0130 00"],
            "pii_spans": [json.dumps([{"start": 0, "end": 27, "label": "iban"}])],
            "language": ["German"],
        },
    )
    [nemotron] = load_nemotron_test(tmp_path)
    [gretel] = load_gretel_test(tmp_path)
    assert (nemotron.language, nemotron.spans) == ("us", (LabelledSpan(0, 6, "email"),))
    assert (gretel.language, gretel.spans) == ("German", (LabelledSpan(0, 27, "iban"),))


@pytest.mark.network
@pytest.mark.data
def test_download_fetches_both_test_files(tmp_path):
    download_structured_datasets(tmp_path)
    assert all((tmp_path / item.relative_path).stat().st_size > 0 for item in STRUCTURED_DATASET_FILES)
