"""Command line entry point: `python -m pii_toolbox.evaluation run --tool <name> --dataset tab --split dev ...`."""

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

from pii_toolbox.core import Dataset, DatasetSplit, DetectorRegistry, IdentityResolver, RunEnvironment, TabMetricsScorer
from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.datasets.mapa import MapaDataset
from pii_toolbox.datasets.tab import TabDataset
from pii_toolbox.detection.registry import FixedDetectorRegistry
from pii_toolbox.evaluation.results_io import write_run
from pii_toolbox.evaluation.run import RunRequest, run_evaluation

DatasetFactory = Callable[[Path], Dataset]

DATASET_FACTORIES: Mapping[str, DatasetFactory] = {"tab": TabDataset, "mapa": MapaDataset}
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


def main(
    argv: Sequence[str] | None = None,
    *,
    registry: DetectorRegistry | None = None,
    dataset_factories: Mapping[str, DatasetFactory] = DATASET_FACTORIES,
    tab_scorer: TabMetricsScorer | None = None,
    resolver: IdentityResolver | None = None,
    environment: RunEnvironment | None = None,
) -> Path:
    """Runs the command and returns the written result file. Collaborators default to the real ones."""
    args = build_parser().parse_args(argv)
    dataset = dataset_factories[args.dataset](args.data_dir or default_data_dir())
    coreference = args.coref == "on"
    if args.dataset == "tab" and tab_scorer is None:
        tab_scorer = _default_tab_scorer()
    if coreference and resolver is None:
        resolver = _default_resolver()
    request = RunRequest(args.tool, dataset, SPLIT_NAMES[args.split], coreference, args.language)
    with resolver if hasattr(resolver, "__enter__") else nullcontext():
        run = run_evaluation(request, registry or FixedDetectorRegistry(), tab_scorer, resolver, environment)
    return write_run(run, args.out)


if __name__ == "__main__":
    main()
