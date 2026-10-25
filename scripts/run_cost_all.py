"""Measures cost for every measured setup, each in a fresh process, and writes one JSON file per setup.

Results go to results/cost/<setup>.json, where `+` in a setup name becomes `__`. Each file holds
peak memory, load seconds, throughput at 1,000, 10,000 and 100,000 characters, median latency
per 1,000 characters, whether the setup fits AWS Lambda's 10 GB, and the environment it ran in.
The language model is not measured because no model has been selected.
"""

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from pii_toolbox.core import Detector, DetectorRegistry
from pii_toolbox.detection.registry import FixedDetectorRegistry, build_from_config
from pii_toolbox.evaluation.cost import SubprocessCostRunner

SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPTS_DIR.parent / "results" / "cost"
INPUT_SIZES = [1000, 10000, 100000]
LAMBDA_MEMORY_BYTES = 10 * 1024**3
PATTERN_DETECTOR = "presidio_structured"
MODEL_DETECTORS = ["spacy_sm", "spacy_lg", "roberta", "gliner"]
NOT_MEASURED = ["language model: no model has been selected yet"]
LIBRARIES = [
    "presidio-analyzer",
    "spacy",
    "gliner",
    "transformers",
    "torch",
    "huggingface_hub",
    "numpy",
    "psutil",
]


def measured_configs() -> list[str]:
    """Presidio alone, each model alone, and each model with Presidio."""
    return (
        [PATTERN_DETECTOR]
        + MODEL_DETECTORS
        + [f"{model}+{PATTERN_DETECTOR}" for model in MODEL_DETECTORS]
    )


class SetupRegistry(DetectorRegistry):
    """Builds a detector or a `model+presidio_structured` union from a setup name."""

    def __init__(self) -> None:
        self._base = FixedDetectorRegistry()

    def names(self) -> frozenset[str]:
        return frozenset(measured_configs())

    def build(self, name: str) -> Detector:
        return build_from_config([name], self._base)[0]


def registry() -> DetectorRegistry:
    return SetupRegistry()


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict:
    versions = {}
    for library in LIBRARIES:
        try:
            versions[library] = metadata.version(library)
        except metadata.PackageNotFoundError:
            continue
    memory = _sysctl("hw.memsize")
    return {
        "machine": _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine(),
        "memory_bytes": int(memory) if memory else None,
        "cpu_cores": _sysctl("hw.ncpu"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "library_versions": versions,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runner = SubprocessCostRunner("run_cost_all:registry", python_path=[SCRIPTS_DIR])
    run_environment = environment()
    for setup in measured_configs():
        raw = runner.run_child(setup, INPUT_SIZES)
        result = {
            "setup": setup,
            "peak_memory_bytes": raw["peak_memory_bytes"],
            "load_seconds": raw["load_seconds"],
            "throughput_chars_per_second": raw["throughput_chars_per_second"],
            "latency_per_1000_chars_median_seconds": raw["latency_per_1000_chars_median_seconds"],
            "fits_lambda": raw["peak_memory_bytes"] < LAMBDA_MEMORY_BYTES,
            "environment": run_environment,
            "not_measured": NOT_MEASURED,
        }
        path = RESULTS_DIR / f"{setup.replace('+', '__')}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{setup}: peak {raw['peak_memory_bytes'] / 1024**3:.2f} GiB, load {raw['load_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
