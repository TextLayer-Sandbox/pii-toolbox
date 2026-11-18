"""Loads one detector, measures its cost over synthetic text and prints the result as JSON."""

import argparse
import importlib
import json
import os
import resource
import statistics
import sys
import time
from collections.abc import Callable, Sequence

from pii_toolbox.core import Detector, DetectorRegistry, Document

TIMED_RUNS = 5
SYNTHETIC_SENTENCE = "Alice Example called Bob Sample on 3 May 2021 about invoice 4471 in Springfield. "


def synthetic_text(size: int) -> str:
    repeats = size // len(SYNTHETIC_SENTENCE) + 1
    return (SYNTHETIC_SENTENCE * repeats)[:size]


def accelerator_memory_bytes() -> int:
    """Memory held by the Apple GPU allocator, which process RSS does not include."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.backends.mps.is_available():
        return 0
    return torch.mps.driver_allocated_memory()


def peak_memory_bytes() -> int:
    """Peak process memory plus whatever the GPU allocator holds at the end of the run."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    process_peak = peak if sys.platform == "darwin" else peak * 1024
    return process_peak + accelerator_memory_bytes()


def load_registry(spec: str) -> DetectorRegistry:
    module_name, attribute = spec.split(":")
    target = getattr(importlib.import_module(module_name), attribute)
    return target() if callable(target) else target


def measure_detector(
    build: Callable[[], Detector],
    sizes: Sequence[int],
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Times loading and detection; only the timed runs and the load read the clock."""
    started = clock()
    detector = build()
    detector.load_models()
    load_seconds = clock() - started

    throughput: dict[str, float] = {}
    per_thousand_latencies: list[float] = []
    for size in sizes:
        document = Document(id=f"synthetic-{size}", text=synthetic_text(size), language="en")
        detector.detect(document)
        durations = []
        for _ in range(TIMED_RUNS):
            started = clock()
            detector.detect(document)
            durations.append(clock() - started)
        median_duration = statistics.median(durations)
        throughput[str(size)] = size / median_duration if median_duration > 0 else float("inf")
        per_thousand_latencies.extend(duration / size * 1000 for duration in durations)

    return {
        "peak_memory_bytes": peak_memory_bytes(),
        "load_seconds": load_seconds,
        "throughput_chars_per_second": throughput,
        "latency_per_1000_chars_median_seconds": statistics.median(per_thousand_latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("detector")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--sizes", required=True)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    sizes = [int(part) for part in args.sizes.split(",")]
    result = measure_detector(lambda: registry.build(args.detector), sizes)
    result["pid"] = os.getpid()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
