"""Runs every evaluation run in the configuration plan and writes one aggregate file for each.

    uv run python scripts/run_all_evaluations.py --split test --out results/test

Each (configuration, dataset, language, co-reference off or on) from all_runs() is scored through
the evaluation command and saved as `<tool>__<dataset>__<language>__<split>__coref-<off|on>.json`,
next to one environment.json. A run that fails is reported at the end and makes the exit code 1;
the remaining runs still execute. `--skip-existing` leaves runs whose file is already in the output
folder alone and `--exclude dataset:language:on|off` leaves a kind of run out. Only aggregate scores are written.
"""

import argparse
import itertools
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pii_toolbox.core import Dataset, DatasetSplit, DetectorRegistry, IdentityResolver, TabMetricsScorer
from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.evaluation.__main__ import DATASET_FACTORIES, DatasetFactory, SPLIT_NAMES, _default_tab_scorer, main as run_one
from pii_toolbox.evaluation.configs import RunSpec, all_runs
from pii_toolbox.evaluation.results_io import ENVIRONMENT_FILE_NAME, run_file_name


@dataclass(frozen=True)
class RunFailure:
    """A run that raised, with the error text."""

    spec: RunSpec
    error: str


class LimitedDataset(Dataset):
    """Wraps a dataset so each load yields at most `limit` documents; used to try the pipeline on a tiny slice."""

    def __init__(self, inner: Dataset, limit: int) -> None:
        self._inner = inner
        self._limit = limit

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def languages(self) -> frozenset[str]:
        return self._inner.languages

    def load(self, split: DatasetSplit, language: str | None = None) -> Iterator:
        return itertools.islice(self._inner.load(split, language), self._limit)


def result_file_name(spec: RunSpec, split: DatasetSplit) -> str:
    """The saved file of a run; the language is part of the name because MAPA is run once per language."""
    base = run_file_name(spec.config, spec.dataset, split, spec.coreference)
    prefix = f"{spec.config}__{spec.dataset}__"
    return f"{prefix}{spec.language}__{base[len(prefix):]}"


def run_all(
    split: DatasetSplit,
    out_dir: Path,
    *,
    specs: Sequence[RunSpec] | None = None,
    max_documents: int | None = None,
    registry: DetectorRegistry | None = None,
    dataset_factories: Mapping[str, DatasetFactory] = DATASET_FACTORIES,
    tab_scorer: TabMetricsScorer | None = None,
    resolver: IdentityResolver | None = None,
    data_dir: Path | None = None,
    report: Callable[[str], None] = print,
) -> list[RunFailure]:
    """Runs every spec into out_dir and returns the failures; every spec is attempted."""
    specs = list(all_runs(registry) if specs is None else specs)
    factories = dict(dataset_factories)
    if max_documents is not None:
        factories = {name: _limited(factory, max_documents) for name, factory in factories.items()}
    if tab_scorer is None and any(spec.dataset == "tab" for spec in specs):
        tab_scorer = _default_tab_scorer()
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for index, spec in enumerate(specs, start=1):
        label = f"{spec.config} {spec.dataset} {spec.language} coref-{'on' if spec.coreference else 'off'}"
        report(f"[{index}/{len(specs)}] {label}")
        try:
            with tempfile.TemporaryDirectory(prefix="pii-run-") as scratch:
                written = run_one(
                    _arguments(spec, split, Path(scratch), data_dir),
                    registry=registry,
                    dataset_factories=factories,
                    tab_scorer=tab_scorer,
                    resolver=resolver,
                )
                shutil.move(written, out_dir / result_file_name(spec, split))
                shutil.move(Path(scratch) / ENVIRONMENT_FILE_NAME, out_dir / ENVIRONMENT_FILE_NAME)
        except Exception as error:
            report(f"  failed: {type(error).__name__}: {error}")
            failures.append(RunFailure(spec, f"{type(error).__name__}: {error}"))
    return failures


def _limited(factory: DatasetFactory, limit: int) -> DatasetFactory:
    return lambda data_dir: LimitedDataset(factory(data_dir), limit)


def _arguments(spec: RunSpec, split: DatasetSplit, scratch: Path, data_dir: Path | None) -> list[str]:
    split_name = next(name for name, value in SPLIT_NAMES.items() if value is split)
    arguments = [
        "run",
        "--tool", spec.config,
        "--dataset", spec.dataset,
        "--split", split_name,
        "--language", spec.language,
        "--coref", "on" if spec.coreference else "off",
        "--out", str(scratch),
    ]
    if data_dir is not None:
        arguments += ["--data-dir", str(data_dir)]
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=("test", "dev"))
    parser.add_argument("--out", type=Path, default=Path("results/test"))
    parser.add_argument("--coref", choices=("both", "off"), default="both", help="`off` skips the co-reference runs")
    parser.add_argument("--max-documents", type=int, default=None, help="score only the first N documents of each load")
    parser.add_argument("--skip-existing", action="store_true", help="do not rerun a run whose result file exists")
    parser.add_argument("--exclude", action="append", default=[], metavar="DATASET:LANGUAGE:on|off", help="leave out matching runs")
    parser.add_argument("--only-config", default=None, help="run a single configuration, such as spacy_sm")
    parser.add_argument("--data-dir", type=Path, default=None, help="default: PII_DATA_DIR, else the user cache")
    args = parser.parse_args(argv)
    split = SPLIT_NAMES[args.split]
    specs = [
        spec
        for spec in all_runs()
        if (args.coref == "both" or not spec.coreference)
        and args.only_config in (None, spec.config)
        and f"{spec.dataset}:{spec.language}:{'on' if spec.coreference else 'off'}" not in args.exclude
        and not (args.skip_existing and (args.out / result_file_name(spec, split)).exists())
    ]
    failures = run_all(
        split, args.out, specs=specs, max_documents=args.max_documents, data_dir=args.data_dir or default_data_dir()
    )
    for failure in failures:
        print(f"FAILED {failure.spec}: {failure.error}", file=sys.stderr)
    print(f"{len(specs) - len(failures)} of {len(specs)} runs written to {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
