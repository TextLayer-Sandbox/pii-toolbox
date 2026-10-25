"""Rebuilds Tables 1 to 6, Charts A to F and a manifest from saved test-split results.

    uv run python scripts/build_outputs.py --results results --out article_outputs

`--results` holds the run result files and a `cost` folder. A result file measured on any split
other than test stops the build and is named in the message. No dataset is read and no model is loaded.
"""

import argparse
import sys
from pathlib import Path

from pii_toolbox.evaluation.build_all import MissingCostResultsError, NonTestResultError, build_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="folder of saved result files, with a cost subfolder")
    parser.add_argument("--out", type=Path, required=True, help="folder to write the tables, charts and manifest to")
    args = parser.parse_args(argv)
    try:
        written = build_outputs(args.results, args.out)
    except (NonTestResultError, MissingCostResultsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(written)} files to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
