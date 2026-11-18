"""Command line entry point: `python -m pii_toolbox.evaluation run --tool <name> --dataset tab --split dev ...`."""

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

from pii_toolbox.core import Dataset, DatasetSplit, DetectorRegistry, IdentityResolver, RunEnvironment, TabMetricsScorer
from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.datasets.mapa import MapaDataset
from pii_toolbox.datasets.tab import TabDataset
from pii_toolbox.datasets.worldwide import WorldwideDataset
from pii_toolbox.detection.registry import FixedDetectorRegistry
from pii_toolbox.evaluation.results_io import ResultRecord, write_run
from pii_toolbox.evaluation.run import RunRequest, run_evaluation

DatasetFactory = Callable[[Path], Dataset]

DATASET_FACTORIES: Mapping[str, DatasetFactory] = {
    "tab": TabDataset,
    "mapa": MapaDataset,
    "worldwide": WorldwideDataset,
}
SPLIT_NAMES: Mapping[str, DatasetSplit] = {
    "train": DatasetSplit.TRAIN,
    "dev": DatasetSplit.DEV,
    "validation": DatasetSplit.DEV,
    "test": DatasetSplit.TEST,
}


def _default_tab_scorer() -> TabMetricsScorer:
    from pii_toolbox.evaluation.tab_metrics import BertInformationWeigher, OfficialTabMetricsScorer, spacy_token_annotator

    return OfficialTabMetricsScorer(spacy_token_annotator(), BertInformationWeigher())


def _default_resolver() -> IdentityResolver:
    from pii_toolbox.coref.resolver import MaverickResolver

    return MaverickResolver()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pii_toolbox.evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="score one detector configuration on one dataset split")
    run.add_argument("--tool", required=True, help="registry name, or `a+b` for a union")
    run.add_argument("--dataset", required=True, choices=sorted(DATASET_FACTORIES))
    run.add_argument("--split", required=True, choices=sorted(SPLIT_NAMES), help="validation is the dev split of MAPA")
    run.add_argument("--coref", required=True, choices=("off", "on"))
    run.add_argument("--out", required=True, type=Path, help="directory for the result and environment files")
    run.add_argument("--language", default=None, help="ISO code to restrict the run to; default every language")
    run.add_argument("--data-dir", type=Path, default=None, help="default: PII_DATA_DIR, else the user cache")
    return parser


def evaluate_one(
    tool: str,
    dataset_name: str,
    split: DatasetSplit,
    coreference: bool,
    language: str | None,
    data_dir: Path | None,
    *,
    registry: DetectorRegistry | None = None,
    dataset_factories: Mapping[str, DatasetFactory] = DATASET_FACTORIES,
    tab_scorer: TabMetricsScorer | None = None,
    resolver: IdentityResolver | None = None,
    environment: RunEnvironment | None = None,
) -> ResultRecord:
    """Scores one detector configuration on one dataset split and returns the record, writing nothing; omitted arguments default to the real ones, the TAB scorer only for TAB and the resolver only with co-reference."""
    dataset = dataset_factories[dataset_name](data_dir or default_data_dir())
    if dataset_name == "tab" and tab_scorer is None:
        tab_scorer = _default_tab_scorer()
    if coreference and resolver is None:
        resolver = _default_resolver()
    request = RunRequest(tool, dataset, split, coreference, language)
    with resolver if hasattr(resolver, "__enter__") else nullcontext():
        return run_evaluation(request, registry or FixedDetectorRegistry(), tab_scorer, resolver, environment)


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: DetectorRegistry | None = None,
    dataset_factories: Mapping[str, DatasetFactory] = DATASET_FACTORIES,
    tab_scorer: TabMetricsScorer | None = None,
    resolver: IdentityResolver | None = None,
    environment: RunEnvironment | None = None,
) -> Path:
    """Runs the command and returns the written result file; omitted keyword arguments default to the real ones."""
    args = build_parser().parse_args(argv)
    record = evaluate_one(
        args.tool,
        args.dataset,
        SPLIT_NAMES[args.split],
        args.coref == "on",
        args.language,
        args.data_dir,
        registry=registry,
        dataset_factories=dataset_factories,
        tab_scorer=tab_scorer,
        resolver=resolver,
        environment=environment,
    )
    return write_run(record, args.out, args.language)


if __name__ == "__main__":
    main()
