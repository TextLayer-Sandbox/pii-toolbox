"""Cost measurement that runs each detector in its own fresh process."""

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from pii_toolbox.core import CostMeasurement, CostRunner


class SubprocessCostRunner(CostRunner):
    """Spawns the cost child for each measurement so peak memory belongs to one detector.

    `registry` is a `module:attr` path to a detector registry (or a callable returning
    one); `python_path` lists extra directories the child can import it from.
    """

    def __init__(self, registry: str, python_path: Sequence[Path] = ()) -> None:
        self._registry = registry
        self._python_path = [str(path) for path in python_path]

    def run_child(self, detector_name: str, input_sizes: Sequence[int]) -> dict:
        """Returns the child's raw JSON result, including its process id."""
        environment = dict(os.environ)
        search_path = self._python_path + [environment.get("PYTHONPATH", "")]
        environment["PYTHONPATH"] = os.pathsep.join(part for part in search_path if part)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pii_toolbox.evaluation.cost_child",
                detector_name,
                "--registry",
                self._registry,
                "--sizes",
                ",".join(str(size) for size in input_sizes),
            ],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        return json.loads(completed.stdout)

    def measure(self, detector_name: str, input_sizes: Sequence[int]) -> CostMeasurement:
        raw = self.run_child(detector_name, input_sizes)
        return CostMeasurement(
            peak_memory_bytes=raw["peak_memory_bytes"],
            load_seconds=raw["load_seconds"],
            throughput_chars_per_second={
                int(size): rate for size, rate in raw["throughput_chars_per_second"].items()
            },
            latency_per_1000_chars_median_seconds=raw["latency_per_1000_chars_median_seconds"],
        )
