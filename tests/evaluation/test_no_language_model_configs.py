import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_PACKAGE = REPO_ROOT / "src" / "pii_toolbox" / "evaluation"
LANGUAGE_MODEL_MENTION = re.compile(r"\bllm\b|model[_ ]pick", re.IGNORECASE)


@pytest.mark.parametrize(
    "removed",
    ["scripts/score_llm_candidates.py", "tests/evaluation/test_llm_shortlist.py"],
)
def test_the_language_model_scoring_files_do_not_exist(removed):
    assert not (REPO_ROOT / removed).exists()


@pytest.mark.parametrize("module", sorted(EVALUATION_PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_no_evaluation_module_mentions_an_llm_detector_or_a_model_pick(module):
    assert not LANGUAGE_MODEL_MENTION.search(module.read_text())
