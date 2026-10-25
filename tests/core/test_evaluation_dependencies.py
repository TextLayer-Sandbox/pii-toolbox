"""Checks that pyproject.toml declares, with a reason comment, each package TAB's evaluation script needs."""

import importlib
import re
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
TAB_PACKAGES = ["spacy", "tqdm", "pandas", "intervaltree", "numpy", "torch", "transformers"]


@pytest.mark.parametrize("package", TAB_PACKAGES)
def test_package_is_declared_with_reason_comment(package):
    pattern = re.compile(rf'^\s*"{package}[^"]*",\s*#\s*\S+', re.MULTILINE)
    assert pattern.search(PYPROJECT.read_text()), f"{package} has no declaration with a reason comment"


@pytest.mark.parametrize("package", TAB_PACKAGES)
def test_package_imports(package):
    importlib.import_module(package)
