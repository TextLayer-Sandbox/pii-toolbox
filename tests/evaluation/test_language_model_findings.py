import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINDINGS = REPO_ROOT / "results" / "language_model_findings.md"
TIMING_DIR = REPO_ROOT / "results" / "language_model_findings"
BEFORE_FIX = TIMING_DIR / "runtime_timing_before_fix.json"
AFTER_FIX = TIMING_DIR / "runtime_timing_after_fix.json"


def test_write_up_states_the_cost_findings():
    text = FINDINGS.read_text()
    for phrase in ("Apple M5", "32 GB", "9.67 s per 1,000 characters", "0.8 s per item", "D17", "too slow"):
        assert phrase in text


def test_timing_files_hold_latency_for_both_runtimes():
    for path in (BEFORE_FIX, AFTER_FIX):
        rows = json.loads(path.read_text())
        assert {row["runtime"] for row in rows} == {"llama.cpp", "mlx"}
        for row in rows:
            assert row["latency_per_1000_chars_seconds"] > 0


def test_timing_after_the_fix_holds_valid_output_rates():
    rows = json.loads(AFTER_FIX.read_text())
    for row in rows:
        assert 0 <= row["valid_structured_output_rate"] <= 1
