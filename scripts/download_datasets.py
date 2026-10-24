"""Command line entry point that downloads TAB and MAPA into PII_DATA_DIR."""

import argparse
from pathlib import Path

from pii_toolbox.datasets.download import default_data_dir, download_datasets, network_fetch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the TAB and MAPA corpora. Skips files already present."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="target directory (default: PII_DATA_DIR, else the user cache)",
    )
    args = parser.parse_args()
    dest = args.dest or default_data_dir()
    fetched = download_datasets(dest, network_fetch)
    print(f"datasets in {dest}: fetched {len(fetched)} files")
    for sub in ("tab", "mapa"):
        count = len(list((dest / sub).glob("*")))
        print(f"  {sub}: {count} files")


if __name__ == "__main__":
    main()
