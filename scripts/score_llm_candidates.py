"""Scores the shortlisted local language models on TAB dev and MAPA validation, and writes one comparison table.

Each candidate is a 4-bit GGUF file run through llama.cpp. A candidate is scored in its own
child process so its peak memory is its own. The only split ever read is dev (MAPA's
validation file); `--split test` is refused. Results hold aggregate numbers only.

    uv run python scripts/score_llm_candidates.py --download   # fetch the model files first
    uv run python scripts/score_llm_candidates.py --split dev
"""

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pii_toolbox.core import (
    Dataset,
    DatasetSplit,
    TabMetricsScorer,
    TabOfficialMetrics,
)
from pii_toolbox.datasets.download import default_data_dir
from pii_toolbox.datasets.mapa import MapaDataset
from pii_toolbox.datasets.tab import TabDataset
from pii_toolbox.detection.llm_detector import LlmDetector, LlmRuntime, build_prompt
from pii_toolbox.detection.llm_runtimes import LlamaCppRuntime
from pii_toolbox.detection.registry import FixedDetectorRegistry
from pii_toolbox.evaluation.model_manifest import default_model_dir
from pii_toolbox.evaluation.results_io import load_for_selection, write_run
from pii_toolbox.evaluation.run import RunRequest, run_evaluation

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "llm_shortlist"
MODEL_FOLDER_NAME = "llm_shortlist"
ALLOWED_SPLIT = "dev"
SCORED_SPLIT = DatasetSplit.DEV
MAPA_LANGUAGES = ("en", "fr", "de")
RUNTIME_NAME = "llama.cpp"
QUANTISATION = "Q4_K_M"


@dataclass(frozen=True)
class Candidate:
    """One language model at one quantisation: where its file comes from and under what licence."""

    name: str
    display_name: str
    repository: str
    revision: str
    file_name: str
    licence: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("qwen3_8b", "Qwen3 8B", "Qwen/Qwen3-8B-GGUF", "7c41481f57cb95916b40956ab2f0b139b296d974", "Qwen3-8B-Q4_K_M.gguf", "Apache-2.0"),
    Candidate("gemma3_12b", "Gemma 3 12B", "ggml-org/gemma-3-12b-it-GGUF", "ec0cbabd8dbff316f659876a50202295c3c4a314", "gemma-3-12b-it-Q4_K_M.gguf", "Gemma Terms of Use"),
    Candidate("mistral_nemo_12b", "Mistral Nemo 12B", "bartowski/Mistral-Nemo-Instruct-2407-GGUF", "a2dd64a0a76ea1bdb2bb6ab6fa5496b003c7c908", "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf", "Apache-2.0"),
    Candidate("llama3_1_8b", "Llama 3.1 8B", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "bf5b95e96dac0462e2a09145ec66cae9a3f12067", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Llama 3.1 Community License"),
)

TABLE_HEADER = (
    "Model",
    "TAB dev F2 strict",
    "TAB dev F2 relaxed",
    "MAPA validation F2 en",
    "MAPA validation F2 fr",
    "MAPA validation F2 de",
    "Valid structured output",
    "Peak memory (GiB)",
    "Load time (s)",
    "Latency per 1,000 chars (s)",
    "Licence",
    "Runtime",
)

RuntimeLoader = Callable[[Candidate], LlmRuntime]
DatasetFactories = Mapping[str, Callable[[Path], Dataset]]


class _SkippedTabMetrics(TabMetricsScorer):
    """The shortlist needs span scores only, so TAB's own metrics are not computed."""

    def score(self, results):
        return TabOfficialMetrics(0.0, 0.0, 0.0)


class _MeasuringRuntime:
    """Wraps a runtime to count calls and add up the time spent and the passage characters sent."""

    def __init__(self, runtime: LlmRuntime) -> None:
        self._runtime = runtime
        self.calls = 0
        self.seconds = 0.0
        self.passage_characters = 0
        self._prompt_overhead = len(build_prompt(""))

    def generate_structured(self, prompt: str, json_schema: dict[str, Any]) -> str:
        started = time.perf_counter()
        reply = self._runtime.generate_structured(prompt, json_schema)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        self.passage_characters += max(len(prompt) - self._prompt_overhead, 0)
        return reply


def model_path(models_dir: Path, candidate: Candidate) -> Path:
    return models_dir / MODEL_FOLDER_NAME / candidate.name / candidate.file_name


def download_candidate(candidate: Candidate, models_dir: Path) -> Path:
    """Fetches the pinned model file into the models directory."""
    from huggingface_hub import hf_hub_download

    target = model_path(models_dir, candidate).parent
    target.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(candidate.repository, candidate.file_name, revision=candidate.revision, local_dir=target))


def _peak_memory_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def score_candidate(
    candidate: Candidate,
    runtime_loader: RuntimeLoader,
    dataset_factories: DatasetFactories,
    data_dir: Path,
    runs_dir: Path,
) -> dict[str, Any]:
    """Loads the model, scores it on TAB dev and MAPA validation, and returns the table row.

    Each scored run is saved under runs_dir and read back through the selection loader, which
    refuses anything but dev or validation results.
    """
    loading_started = time.perf_counter()
    runtime = _MeasuringRuntime(runtime_loader(candidate))
    load_seconds = time.perf_counter() - loading_started
    detector = LlmDetector(runtime, name=candidate.name)
    registry = FixedDetectorRegistry(factory_overrides={candidate.name: lambda: detector})

    def scored_run(dataset_name: str, language: str | None):
        request = RunRequest(candidate.name, dataset_factories[dataset_name](data_dir), SCORED_SPLIT, False, language)
        record = run_evaluation(request, registry, _SkippedTabMetrics())
        folder = runs_dir / candidate.name / f"{dataset_name}-{language or 'all'}"
        return load_for_selection(write_run(record, folder))

    tab = scored_run("tab", None)
    mapa_f2 = {language: scored_run("mapa", language).exact_boundaries.overall_micro.f2 for language in MAPA_LANGUAGES}
    valid_calls = runtime.calls - detector.failed_calls
    return {
        "name": candidate.name,
        "model": candidate.display_name,
        "repository": candidate.repository,
        "revision": candidate.revision,
        "file": candidate.file_name,
        "quantisation": QUANTISATION,
        "tab_dev_f2_strict": tab.exact_boundaries.overall_micro.f2,
        "tab_dev_f2_relaxed": tab.any_overlap.overall_micro.f2,
        "mapa_validation_f2": mapa_f2,
        "valid_structured_output_rate": valid_calls / runtime.calls if runtime.calls else 0.0,
        "model_calls": runtime.calls,
        "peak_memory_bytes": _peak_memory_bytes(),
        "load_seconds": load_seconds,
        "latency_per_1000_chars_seconds": runtime.seconds / runtime.passage_characters * 1000 if runtime.passage_characters else 0.0,
        "licence": candidate.licence,
        "runtime": RUNTIME_NAME,
    }


def render_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Language model shortlist, dev splits",
        "",
        "Scores are F2 with strict matching unless marked relaxed, on TAB dev and MAPA validation. "
        "This table names no winner.",
        "",
        "| " + " | ".join(TABLE_HEADER) + " |",
        "| " + " | ".join("---" for _ in TABLE_HEADER) + " |",
    ]
    for row in rows:
        cells = [
            row["model"],
            f"{row['tab_dev_f2_strict']:.3f}",
            f"{row['tab_dev_f2_relaxed']:.3f}",
            *(f"{row['mapa_validation_f2'][language]:.3f}" for language in MAPA_LANGUAGES),
            f"{row['valid_structured_output_rate']:.3f}",
            f"{row['peak_memory_bytes'] / 1024**3:.2f}",
            f"{row['load_seconds']:.1f}",
            f"{row['latency_per_1000_chars_seconds']:.2f}",
            row["licence"],
            f"{row['runtime']} ({row['quantisation']})",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Model files:", ""]
    lines += [f"- {row['model']}: {row['repository']} at {row['revision']}, file {row['file']}" for row in rows]
    return "\n".join(lines) + "\n"


def write_comparison(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
    (out_dir / "comparison.md").write_text(render_markdown(rows), encoding="utf-8")


def _score_in_child(candidate: Candidate, models_dir: Path, data_dir: Path, runs_dir: Path) -> dict[str, Any]:
    """Runs one candidate in a fresh Python process and returns its row."""
    with tempfile.TemporaryDirectory() as temporary:
        row_file = Path(temporary) / "row.json"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--split", ALLOWED_SPLIT, "--only", candidate.name,
            "--row-out", str(row_file), "--models-dir", str(models_dir), "--data-dir", str(data_dir), "--runs-dir", str(runs_dir),
        ]
        subprocess.run(command, check=True)
        return json.loads(row_file.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=[ALLOWED_SPLIT], default=ALLOWED_SPLIT, help="only dev is allowed")
    parser.add_argument("--download", action="store_true", help="fetch the model files and stop")
    parser.add_argument("--only", choices=[c.name for c in CANDIDATES], help="score one candidate in this process")
    parser.add_argument("--row-out", type=Path, help="with --only: where to write the candidate's row")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=RESULTS_DIR / "runs")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    return parser


def _llama_cpp_loader(models_dir: Path) -> RuntimeLoader:
    return lambda candidate: LlamaCppRuntime(model_path(models_dir, candidate))


def main(
    argv: Sequence[str] | None = None,
    *,
    candidates: Sequence[Candidate] = CANDIDATES,
    runtime_loader: RuntimeLoader | None = None,
    dataset_factories: DatasetFactories | None = None,
    score_one: Callable[[Candidate, Path, Path, Path], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Scores the candidates (each via `score_one`, a child process by default) and writes the comparison."""
    args = build_parser().parse_args(argv)
    models_dir = args.models_dir or default_model_dir()
    data_dir = args.data_dir or default_data_dir()
    if args.download:
        for candidate in candidates:
            print(f"downloaded {download_candidate(candidate, models_dir)}")
        return []
    factories = dataset_factories or {"tab": TabDataset, "mapa": MapaDataset}
    if args.only:
        candidate = next(c for c in candidates if c.name == args.only)
        loader = runtime_loader or _llama_cpp_loader(models_dir)
        row = score_candidate(candidate, loader, factories, data_dir, args.runs_dir)
        args.row_out.write_text(json.dumps(row), encoding="utf-8")
        return [row]
    measure = score_one or _score_in_child
    rows = []
    for candidate in candidates:
        rows.append(measure(candidate, models_dir, data_dir, args.runs_dir))
        print(f"scored {candidate.display_name}", flush=True)
    write_comparison(rows, args.out)
    return rows


if __name__ == "__main__":
    main()
