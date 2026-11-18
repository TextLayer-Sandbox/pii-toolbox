"""Locates the directory that holds the downloaded models."""

import os
from pathlib import Path

MODEL_DIR_ENV_VAR = "PII_MODEL_DIR"


def default_model_dir() -> Path:
    configured = os.environ.get(MODEL_DIR_ENV_VAR)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "pii-toolbox" / "models"
