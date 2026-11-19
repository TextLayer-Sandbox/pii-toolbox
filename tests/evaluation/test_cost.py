import textwrap

from pii_toolbox.core import Detector, Document, PiiType
from pii_toolbox.evaluation.cost import SubprocessCostRunner
from pii_toolbox.evaluation.cost_child import measure_detector

STUB_REGISTRY = textwrap.dedent(
    """
    from pii_toolbox.core import Detector, DetectorRegistry, PiiType


    class StubDetector(Detector):
        def __init__(self, name):
            self._name = name

        name = property(lambda self: self._name)
        languages = property(lambda self: None)
        detects = property(lambda self: frozenset({PiiType.PERSON}))

        def load_models(self):
            pass

        def detect(self, document):
            return []


    class StubRegistry(DetectorRegistry):
        def names(self):
            return frozenset({"alpha", "beta"})

        def build(self, name):
            return StubDetector(name)


    registry = StubRegistry
    """
)

SIZES = [1000, 10000, 100000]


class CountingDetector(Detector):
    def __init__(self):
        self.detect_calls = 0

    @property
    def name(self) -> str:
        return "counting"

    @property
    def languages(self) -> frozenset[str] | None:
        return None

    @property
    def detects(self) -> frozenset[PiiType]:
        return frozenset()

    def load_models(self):
        pass

    def detect(self, document: Document):
        self.detect_calls += 1
        return []


def make_runner(tmp_path):
    (tmp_path / "stub_registry.py").write_text(STUB_REGISTRY)
    return SubprocessCostRunner("stub_registry:registry", python_path=[tmp_path])


def test_measure_reports_every_field(tmp_path):
    measurement = make_runner(tmp_path).measure("alpha", SIZES)
    assert measurement.peak_memory_bytes > 0
    assert measurement.load_seconds >= 0
    assert set(measurement.throughput_chars_per_second) == set(SIZES)
    assert all(rate > 0 for rate in measurement.throughput_chars_per_second.values())
    assert measurement.latency_per_1000_chars_median_seconds >= 0


def test_each_detector_runs_in_its_own_process(tmp_path):
    runner = make_runner(tmp_path)
    first = runner.run_child("alpha", [1000])
    second = runner.run_child("beta", [1000])
    assert first["pid"] != second["pid"]


def test_child_json_has_all_fields(tmp_path):
    raw = make_runner(tmp_path).run_child("alpha", [1000])
    assert {
        "peak_memory_bytes",
        "load_seconds",
        "throughput_chars_per_second",
        "latency_per_1000_chars_median_seconds",
    } <= set(raw)


def test_warm_up_is_excluded_and_median_of_five_is_taken():
    detector = CountingDetector()
    ticks = iter([0.0, 0.5] + [value for duration in [5, 1, 4, 2, 3] for value in (0.0, float(duration))])
    result = measure_detector(lambda: detector, [1000], clock=lambda: next(ticks))
    assert detector.detect_calls == 6
    assert result["load_seconds"] == 0.5
    assert result["throughput_chars_per_second"]["1000"] == 1000 / 3
    assert result["latency_per_1000_chars_median_seconds"] == 3


def test_gpu_allocator_memory_is_added_to_peak_memory(monkeypatch):
    import types

    from pii_toolbox.evaluation import cost_child

    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
        mps=types.SimpleNamespace(driver_allocated_memory=lambda: 5_000_000_000),
    )
    without_gpu = cost_child.peak_memory_bytes()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    assert cost_child.peak_memory_bytes() - without_gpu >= 4_000_000_000
