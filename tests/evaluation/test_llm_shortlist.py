import importlib.util
import json
from pathlib import Path

import pytest

from pii_toolbox.core import AnnotatedDocument, AnswerSpan, Dataset, DatasetSplit, Document, PiiType

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "score_llm_candidates.py"
CANARY = "ZXQCANARYTOKEN"
TEXT = f"Anna Testwell wrote from Exampleton. {CANARY} stays private."
REQUESTED_SPLITS: list[DatasetSplit] = []


def load_script():
    spec = importlib.util.spec_from_file_location("score_llm_candidates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()


class StubDataset(Dataset):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def languages(self):
        return frozenset({"en", "fr", "de"})

    def load(self, split, language=None):
        REQUESTED_SPLITS.append(split)
        answers = (AnswerSpan(0, 13, PiiType.PERSON, "PERSON", None, None, None),)
        for lang in ("en", "fr", "de"):
            if language in (None, lang):
                yield AnnotatedDocument(Document(f"{lang}1", TEXT, lang), answers)


class StubRuntime:
    """Reports the person on every call, but replies with broken JSON on every third call."""

    def __init__(self):
        self.calls = 0

    def generate_structured(self, prompt, json_schema):
        self.calls += 1
        if self.calls % 3 == 0:
            return "not json"
        return json.dumps({"items": [{"text": "Anna Testwell", "type": "PERSON"}]})


FACTORIES = {"tab": lambda d: StubDataset("TAB"), "mapa": lambda d: StubDataset("MAPA")}


def run_main(tmp_path, *extra):
    candidates = script.CANDIDATES[:2]

    def score_one(candidate, models_dir, data_dir, runs_dir):
        return script.score_candidate(candidate, lambda c: StubRuntime(), FACTORIES, data_dir, runs_dir)

    argv = ["--split", "dev", "--out", str(tmp_path / "out"), "--runs-dir", str(tmp_path / "runs"), *extra]
    return script.main(argv, candidates=candidates, dataset_factories=FACTORIES, score_one=score_one)


def test_test_split_is_refused(tmp_path):
    with pytest.raises(SystemExit) as refused:
        script.main(["--split", "test", "--out", str(tmp_path)])
    assert refused.value.code != 0


def test_only_the_dev_split_is_ever_requested(tmp_path):
    REQUESTED_SPLITS.clear()
    run_main(tmp_path)
    assert REQUESTED_SPLITS and set(REQUESTED_SPLITS) == {DatasetSplit.DEV}


def test_script_source_never_names_the_test_split():
    assert "DatasetSplit.TEST" not in SCRIPT.read_text()


def test_comparison_has_a_row_per_candidate_and_every_column(tmp_path):
    rows = run_main(tmp_path)
    markdown = (tmp_path / "out" / "comparison.md").read_text()
    header = next(line for line in markdown.splitlines() if line.startswith("| Model"))
    for column in script.TABLE_HEADER:
        assert column in header
    assert [row["name"] for row in rows] == [c.name for c in script.CANDIDATES[:2]]
    saved = json.loads((tmp_path / "out" / "comparison.json").read_text())
    assert [row["model"] for row in saved] == ["Qwen3 8B", "Gemma 3 12B"]


def test_valid_output_rate_counts_unparseable_replies(tmp_path):
    row = run_main(tmp_path)[0]
    assert 0.0 < row["valid_structured_output_rate"] < 1.0
    assert row["model_calls"] > 0
    assert row["peak_memory_bytes"] > 0


def test_results_hold_no_dataset_text_and_name_no_winner(tmp_path):
    run_main(tmp_path)
    for path in (tmp_path / "out").iterdir():
        assert CANARY not in path.read_text()
    for path in (tmp_path / "runs").rglob("*.json"):
        assert CANARY not in path.read_text()
    assert "winner" not in (tmp_path / "out" / "comparison.md").read_text().lower().replace("names no winner", "")


def test_every_candidate_states_a_quantisation_licence_and_a_pinned_file():
    for candidate in script.CANDIDATES:
        assert candidate.file_name.endswith("Q4_K_M.gguf")
        assert len(candidate.revision) == 40
        assert candidate.licence
