"""Loads a nickname list: a mapping from each lower-case nickname to its one standard given name.

A list file has two comma-separated columns, nickname then standard given name, and the shape
is star-shaped: nicknames link to each other only through their standard name. The loader
refuses a file where one nickname has two different standard names, or where a name is both a
standard name and a nickname of another.
"""

import csv
from pathlib import Path

DEFAULT_NICKNAME_LIST_PATH = Path(__file__).with_name("nicknames.csv")


def load_nickname_list(path: Path) -> dict[str, str]:
    """Return {nickname: standard name} read from the two-column file at `path`.

    Raises ValueError when a row is malformed, a nickname maps to two different standard names,
    or a name is both a standard name and a nickname of another name.
    """
    mapping: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != 2 or not all(cell.strip() for cell in row):
                raise ValueError(f"{path}:{line_number}: expected 'nickname,standard name', got {row!r}")
            nickname, standard = (cell.strip().lower() for cell in row)
            if nickname == standard:
                raise ValueError(f"{path}:{line_number}: {nickname!r} is mapped to itself")
            if mapping.get(nickname, standard) != standard:
                raise ValueError(
                    f"{path}:{line_number}: {nickname!r} maps to both {mapping[nickname]!r} and {standard!r}"
                )
            mapping[nickname] = standard
    for standard in sorted(set(mapping.values()) & set(mapping)):
        raise ValueError(f"{path}: {standard!r} is both a standard name and a nickname of {mapping[standard]!r}")
    return mapping
