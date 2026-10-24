"""Command line entry point that downloads every model in the manifest into PII_MODEL_DIR."""

import argparse
from pathlib import Path

from pii_toolbox.evaluation.model_manifest import (
    default_model_dir,
    download_models,
    network_fetch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the models the toolbox loads. Skips models already present."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="target directory (default: PII_MODEL_DIR, else the user cache)",
    )
    args = parser.parse_args()
    dest = args.dest or default_model_dir()
    fetched = download_models(dest, network_fetch)
    print(f"models in {dest}: fetched {len(fetched)}")


if __name__ == "__main__":
    main()
