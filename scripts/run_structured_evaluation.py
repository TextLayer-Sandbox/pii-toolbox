"""Downloads the two structured-PII datasets, measures Presidio on their test splits and writes aggregate results.

    uv run python scripts/run_structured_evaluation.py --out results/structured

Every predefined Presidio pattern recogniser runs on every document, spread over worker processes.
One `<dataset>__test.json` per dataset is written, holding counts and rates per kind and per group, never
document text or labelled values. `--max-documents` limits each dataset to its first N documents for a timing sample.
"""

import argparse
import itertools
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path

from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.datasets.structured_pii import (
    GRETEL_REVISION,
    NEMOTRON_REVISION,
    download_structured_datasets,
    load_gretel_test,
    load_nemotron_test,
)
from pii_toolbox.evaluation.structured import build_recognisers, build_result, count_documents_parallel

DATASETS = {
    "nemotron": (load_nemotron_test, NEMOTRON_REVISION),
    "gretel": (load_gretel_test, GRETEL_REVISION),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/structured"))
    parser.add_argument("--max-documents", type=int, default=None, help="score only the first N documents of each dataset")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="worker processes (default: every core)")
    parser.add_argument("--data-dir", type=Path, default=None, help="default: PII_DATA_DIR, else the user cache")
    args = parser.parse_args(argv)
    data_dir = args.data_dir or default_data_dir()
    download_structured_datasets(data_dir)
    recogniser_names = sorted(type(recogniser).__name__ for recogniser in build_recognisers())
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (load, revision) in DATASETS.items():
        started = time.perf_counter()
        documents = load(data_dir)
        if args.max_documents is not None:
            documents = itertools.islice(documents, args.max_documents)
        counts = count_documents_parallel(name, documents, args.workers)
        result = build_result(name, revision, recogniser_names, counts)
        result["presidio_analyzer_version"] = metadata.version("presidio-analyzer")
        target = args.out / f"{name}__test.json"
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: {time.perf_counter() - started:.1f} s, written to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
