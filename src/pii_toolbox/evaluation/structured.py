"""Measures how often Presidio's predefined recognisers find labelled structured PII, per kind.

Every pattern and checksum recogniser shipped with the installed presidio-analyzer is switched
on, whatever its country or language, and run directly on each document text; no NLP engine or
language model is involved. A labelled item counts as found when any finding overlaps it, and as
found as the correct kind when a finding whose entity belongs to the item's kind overlaps it. Items of
kinds with a checksum are also split into validly formed and not, and rates are kept for all items and for
valid ones only. Only counts and rates are produced, never document text or labelled values.
"""

import inspect
import itertools
import multiprocessing
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from pii_toolbox.datasets.structured_pii import StructuredDocument
from pii_toolbox.evaluation.structured_kinds import (
    GROUPS,
    KIND_ENTITIES,
    KIND_GROUP,
    KINDS,
    VALIDITY_CHECKS,
    is_validly_formed,
    kind_of_label,
)

_COUNT_FIELDS = ("items", "any_all", "correct_all", "valid_items", "any_valid", "correct_valid")


@dataclass(frozen=True)
class Finding:
    """A Presidio finding: entity name and a character range with an exclusive end."""

    entity: str
    start: int
    end: int


def build_recognisers() -> list:
    """One instance of every predefined pattern or phone recogniser, with all phone regions on."""
    import phonenumbers
    import presidio_analyzer.predefined_recognizers as predefined
    import tldextract
    from presidio_analyzer import PatternRecognizer

    # An empty suffix-list URL tuple makes tldextract use its bundled list instead of downloading one.
    offline_domain_parser = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

    class OfflineEmailRecognizer(predefined.EmailRecognizer):
        def validate_result(self, pattern_text: str) -> bool:
            return offline_domain_parser(pattern_text).fqdn != ""

    recognisers = []
    for name, recogniser_class in sorted(vars(predefined).items()):
        if not inspect.isclass(recogniser_class) or recogniser_class is predefined.EmailRecognizer:
            continue
        if recogniser_class is predefined.PhoneRecognizer:
            recognisers.append(recogniser_class(supported_regions=sorted(phonenumbers.SUPPORTED_REGIONS)))
        elif issubclass(recogniser_class, PatternRecognizer):
            recognisers.append(recogniser_class())
    recognisers.append(OfflineEmailRecognizer())
    return recognisers


def find_entities(recognisers: list, text: str) -> list[Finding]:
    """Every non-empty finding of every recogniser on the text."""
    findings = []
    for recogniser in recognisers:
        for result in recogniser.analyze(text, list(recogniser.supported_entities)):
            if result.start < result.end:
                findings.append(Finding(result.entity_type, result.start, result.end))
    return findings


def item_outcome(kind: str, start: int, end: int, findings: Iterable[Finding]) -> tuple[bool, bool]:
    """Whether the item overlaps any finding, and whether it overlaps a finding of its own kind."""
    entities = KIND_ENTITIES[kind]
    found_any = False
    for finding in findings:
        if finding.start < end and start < finding.end:
            found_any = True
            if finding.entity in entities:
                return True, True
    return found_any, False


def _empty_counts() -> dict[str, int]:
    return {field: 0 for field in _COUNT_FIELDS}


def count_documents(
    dataset: str, documents: Iterable[StructuredDocument], recognisers: list
) -> dict[str, dict[str, dict[str, int]]]:
    """Counts per language and kind (`ALL` is every language) of items and how many were found.

    Items of a kind without a checksum are all valid, so the valid counts equal the all counts.
    """
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for document in documents:
        scored = [
            (kind, span)
            for span in document.spans
            if (kind := kind_of_label(dataset, span.label)) is not None
        ]
        if not scored:
            continue
        findings = find_entities(recognisers, document.text)
        for kind, span in scored:
            found_any, found_correct = item_outcome(kind, span.start, span.end, findings)
            valid = is_validly_formed(kind, document.text[span.start : span.end].strip())
            for language in (document.language, "ALL"):
                bucket = counts.setdefault(language, {}).setdefault(kind, _empty_counts())
                bucket["items"] += 1
                bucket["any_all"] += found_any
                bucket["correct_all"] += found_correct
                if valid:
                    bucket["valid_items"] += 1
                    bucket["any_valid"] += found_any
                    bucket["correct_valid"] += found_correct
    return counts


def merge_counts(
    total: dict[str, dict[str, dict[str, int]]], part: dict[str, dict[str, dict[str, int]]]
) -> None:
    """Adds the counts of part into total."""
    for language, by_kind in part.items():
        for kind, counts in by_kind.items():
            bucket = total.setdefault(language, {}).setdefault(kind, _empty_counts())
            for field in _COUNT_FIELDS:
                bucket[field] += counts[field]


_worker_recognisers: list = []


def _init_worker() -> None:
    _worker_recognisers[:] = build_recognisers()


def _count_chunk(job: tuple[str, list[StructuredDocument]]) -> dict[str, dict[str, dict[str, int]]]:
    dataset, documents = job
    return count_documents(dataset, documents, _worker_recognisers)


def _chunks(documents: Iterable[StructuredDocument], size: int) -> Iterator[list[StructuredDocument]]:
    iterator = iter(documents)
    while chunk := list(itertools.islice(iterator, size)):
        yield chunk


def count_documents_parallel(
    dataset: str, documents: Iterable[StructuredDocument], workers: int, chunk_size: int = 200
) -> dict[str, dict[str, dict[str, int]]]:
    """Counts per language and kind as for one process, spreading chunks of documents over worker processes that each build the recognisers once."""
    total: dict[str, dict[str, dict[str, int]]] = {}
    jobs = ((dataset, chunk) for chunk in _chunks(documents, chunk_size))
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for part in pool.imap_unordered(_count_chunk, jobs):
            merge_counts(total, part)
    return total


def _rate(found: int, total: int) -> float | None:
    return found / total if total else None


def _with_rates(counts: dict[str, int]) -> dict[str, float | int | None]:
    return {
        **counts,
        "any_rate_all": _rate(counts["any_all"], counts["items"]),
        "correct_rate_all": _rate(counts["correct_all"], counts["items"]),
        "any_rate_valid": _rate(counts["any_valid"], counts["valid_items"]),
        "correct_rate_valid": _rate(counts["correct_valid"], counts["valid_items"]),
    }


def summarise(counts: dict[str, dict[str, int]]) -> dict[str, dict]:
    """Per-kind and per-group counts and rates for one language bucket; kinds without items are kept with zero counts."""
    per_kind = {kind: counts.get(kind, _empty_counts()) for kind in KINDS}
    per_group = {}
    for group, kinds in GROUPS.items():
        per_group[group] = {field: sum(per_kind[kind][field] for kind in kinds) for field in _COUNT_FIELDS}
    return {
        "kinds": {
            kind: {
                "group": KIND_GROUP[kind],
                "has_checksum": kind in VALIDITY_CHECKS,
                "presidio_entities": sorted(KIND_ENTITIES[kind]),
                **_with_rates(per_kind[kind]),
            }
            for kind in KINDS
        },
        "groups": {group: _with_rates(values) for group, values in per_group.items()},
    }


def build_result(dataset: str, revision: str, recogniser_names: list[str], counts: dict) -> dict:
    """The saved record of one dataset's run: provenance and summaries overall and per language."""
    return {
        "dataset": dataset,
        "split": "test",
        "revision": revision,
        "recognisers": recogniser_names,
        "overall": summarise(counts.get("ALL", {})),
        "by_language": {
            language: summarise(by_kind) for language, by_kind in sorted(counts.items()) if language != "ALL"
        },
    }
