import importlib
import inspect

import pytest

from pii_toolbox.detection.registry import FixedDetectorRegistry


def test_registry_has_no_llm_detector():
    assert "llm" not in FixedDetectorRegistry().names()


def test_registry_accepts_no_language_model_settings():
    parameters = inspect.signature(FixedDetectorRegistry).parameters
    assert "llm_model_path" not in parameters
    assert "llm_runtime" not in parameters


@pytest.mark.parametrize("module", ["llm_detector", "llm_runtimes"])
def test_language_model_modules_cannot_be_imported(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"pii_toolbox.detection.{module}")
