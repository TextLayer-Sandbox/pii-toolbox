"""Measures identity linking on the TAB documents for each detector configuration.

    uv run python scripts/run_coref_linking.py --split test --out results/coref_linking
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pii_toolbox.coref.resolver import MaverickResolver
from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.datasets.tab import TabDataset
from pii_toolbox.detection.registry import build_from_config
from pii_toolbox.evaluation.__main__ import SPLIT_NAMES
from pii_toolbox.evaluation.coref_linking import linking_report, measure_linking

CONFIGS = ("gliner+presidio_structured", "spacy_lg+presidio_structured")

_SUMMARY_ROWS = (
    ("Entities with 2+ detected mentions and 2+ surface forms: all mentions under one placeholder", "multi_form_entities_linked", "multi_form_entities"),
    ("Entities with 2+ detected mentions: all mentions under one placeholder", "multi_mention_entities_linked", "multi_mention_entities"),
    ("Placeholders holding mentions of 2+ different people", "wrongly_merged_placeholders", "placeholders"),
)


def result_file_name(config: str, split: str) -> str:
    return f"{config}__tab__{split}.json"


def _format_cell(counts: dict, numerator: str, denominator: str) -> str:
    total = counts[denominator]
    rate = f"{counts[numerator] / total:.3f}" if total else "n/a"
    return f"{rate} ({counts[numerator]}/{total})"


def summary_markdown(split: str, reports: dict[str, dict]) -> str:
    """Pooled counts per configuration, resolver off and on."""
    lines = [f"# Identity linking on TAB {split}", ""]
    lines.append(
        "Pooled over annotators (counts summed, rate = numerator/denominator). Detected PERSON spans are matched "
        "to the same annotator's DIRECT or QUASI PERSON mentions by overlap; spans matching none are excluded."
    )
    for config, report in reports.items():
        off, on = report["coref_off"]["pooled"], report["coref_on"]["pooled"]
        lines += ["", f"## {config}", "", "| Measure | Resolver off | Resolver on |", "| --- | --- | --- |"]
        for title, numerator, denominator in _SUMMARY_ROWS:
            lines.append(f"| {title} | {_format_cell(off, numerator, denominator)} | {_format_cell(on, numerator, denominator)} |")
        lines.append(
            f"| Detected PERSON spans matched / unmatched | {off['matched_spans']} / {off['unmatched_spans']} (summed over annotators) "
            f"| {on['matched_spans']} / {on['unmatched_spans']} (summed over annotators) |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=("test", "dev"))
    parser.add_argument("--out", type=Path, default=Path("results/coref_linking"))
    parser.add_argument("--only-config", default=None, choices=CONFIGS, help="run a single configuration")
    parser.add_argument("--data-dir", type=Path, default=None, help="default: PII_DATA_DIR, else the user cache")
    args = parser.parse_args(argv)
    configs = [config for config in CONFIGS if args.only_config in (None, config)]
    dataset = TabDataset(args.data_dir or default_data_dir())
    args.out.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    with MaverickResolver() as resolver:
        for config in configs:
            print(f"{config}", flush=True)
            detector = build_from_config([config])[0]
            detector.load_models()
            measurement = measure_linking(dataset.load(SPLIT_NAMES[args.split]), detector, resolver)
            reports[config] = {
                "config": config,
                "dataset": "tab",
                "split": args.split,
                "coref_off": linking_report(measurement.off),
                "coref_on": linking_report(measurement.on),
            }
            path = args.out / result_file_name(config, args.split)
            path.write_text(json.dumps(reports[config], indent=2) + "\n", encoding="utf-8")
            print(f"  wrote {path}", flush=True)
    (args.out / "summary.md").write_text(summary_markdown(args.split, reports), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
