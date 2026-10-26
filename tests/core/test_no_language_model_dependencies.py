"""Checks that neither the project file nor the lock file names a language-model runtime or its dependency group."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_MODEL_PACKAGES = ["llama-cpp-python", "mlx-lm"]


def test_pyproject_has_no_language_model_group_or_package():
    text = (ROOT / "pyproject.toml").read_text()
    groups = tomllib.loads(text)["dependency-groups"]
    assert "llm" not in groups
    for package in LANGUAGE_MODEL_PACKAGES:
        assert package not in text


def test_lock_lists_no_language_model_package():
    packages = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    names = {package["name"] for package in packages}
    assert not names.intersection(LANGUAGE_MODEL_PACKAGES)
