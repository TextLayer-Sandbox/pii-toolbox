"""Rebuilds Tables 1 to 6, Charts A to F and a manifest from saved test-split results.

    uv run python scripts/build_outputs.py --results results/test --cost results/cost --out results/outputs

`--results` holds the run result files. `--cost` holds the cost result files and defaults to the
`cost` subfolder of `--results`. A result file measured on any split
other than test stops the build and is named in the message. No dataset is read and no model is loaded.
"""

import argparse
import sys
from pathlib import Path

from pii_toolbox.evaluation.build_all import MissingCostResultsError, NonTestResultError, build_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="folder of saved result files")
    parser.add_argument("--cost", type=Path, default=None, help="folder of cost result files (default: the cost subfolder of --results)")
    parser.add_argument("--out", type=Path, required=True, help="folder to write the tables, charts and manifest to")
    args = parser.parse_args(argv)
    try:
        written = build_outputs(args.results, args.out, args.cost)
    except (NonTestResultError, MissingCostResultsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(written)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
