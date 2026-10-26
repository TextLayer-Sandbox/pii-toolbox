import importlib.util
import json
from pathlib import Path

import pytest

from pii_toolbox.core import (
    AnnotatedDocument,
    AnswerSpan,
    Dataset,
    DatasetSplit,
    Detector,
    Document,
    EntityGroup,
    IdentityResolver,
    PiiType,
    ReidentificationRisk,
    Span,
    TabMetricsScorer,
    TabOfficialMetrics,
)
from pii_toolbox.detection.registry import FixedDetectorRegistry
from pii_toolbox.evaluation.configs import RunSpec
from pii_toolbox.evaluation.results_io import ENVIRONMENT_FILE_NAME

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_all_evaluations.py"
_spec = importlib.util.spec_from_file_location("run_all_evaluations", SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

TEXT = "Anna Testwell met Anna there."
FULL_NAME = (0, 13)
SHORT_NAME = (18, 22)


class StubDetector(Detector):
    name = "stub"
    languages = frozenset({"en"})
    detects = frozenset({PiiType.PERSON})

    def load_models(self) -> None:
        pass

    def detect(self, document):
        return [Span(*FULL_NAME, PiiType.PERSON, "PERSON", ("stub",), None)]


class StubResolver(IdentityResolver):
    name = "stub_resolver"

    def resolve(self, document, spans):
        added = Span(*SHORT_NAME, PiiType.PERSON, "PERSON", ("stub_resolver",), None)
        return [EntityGroup((spans[0], added), spans[0])]


class StubTabScorer(TabMetricsScorer):
    def score(self, results):
        return TabOfficialMetrics(0.5, 0.25, 0.75)


class StubDataset(Dataset):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def languages(self):
        return frozenset({"en", "fr"})

    def load(self, split, language=None):
        answers = (
            AnswerSpan(*FULL_NAME, PiiType.PERSON, "PERSON", "A", ReidentificationRisk.DIRECT, "e1"),
            AnswerSpan(*SHORT_NAME, PiiType.PERSON, "PERSON", "A", ReidentificationRisk.DIRECT, "e1"),
        )
        for doc_id, lang in (("d1", "en"), ("d2", "fr")):
            if language in (None, lang):
                yield AnnotatedDocument(Document(doc_id, TEXT, lang), answers)


def stub_registry():
    return FixedDetectorRegistry(factory_overrides={"stub": StubDetector})


FACTORIES = {"tab": lambda d: StubDataset("TAB"), "mapa": lambda d: StubDataset("MAPA")}


def run(tmp_path, specs, **extra):
    return script.run_all(
        DatasetSplit.TEST,
        tmp_path,
        specs=specs,
        registry=stub_registry(),
        dataset_factories=FACTORIES,
        tab_scorer=StubTabScorer(),
        resolver=StubResolver(),
        report=lambda line: None,
        **extra,
    )


def test_every_spec_gets_its_own_file_and_one_environment_file(tmp_path):
    specs = [
        RunSpec("stub", "tab", "en", False),
        RunSpec("stub", "tab", "en", True),
        RunSpec("stub", "mapa", "en", False),
        RunSpec("stub", "mapa", "fr", False),
    ]
    assert run(tmp_path, specs) == []
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [
            ENVIRONMENT_FILE_NAME,
            "stub__tab__en__test__coref-off.json",
            "stub__tab__en__test__coref-on.json",
            "stub__mapa__en__test__coref-off.json",
            "stub__mapa__fr__test__coref-off.json",
        ]
    )


def test_a_failing_run_is_reported_and_the_others_still_run(tmp_path):
    specs = [RunSpec("missing_tool", "tab", "en", False), RunSpec("stub", "tab", "en", False)]
    failures = run(tmp_path, specs)
    assert [failure.spec.config for failure in failures] == ["missing_tool"]
    assert (tmp_path / "stub__tab__en__test__coref-off.json").exists()


def test_the_document_limit_scores_fewer_documents(tmp_path):
    run(tmp_path, [RunSpec("stub", "mapa", "en", False)], max_documents=0)
    result = json.loads((tmp_path / "stub__mapa__en__test__coref-off.json").read_text())
    assert result["exact_boundaries"]["by_language"] == {}


def test_result_names_are_unique_across_the_whole_plan():
    names = [script.result_file_name(spec, DatasetSplit.TEST) for spec in script.all_runs()]
    assert len(set(names)) == len(names)


def test_exclude_and_skip_existing_leave_runs_out(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(script, "run_all", lambda split, out, specs, **kw: seen.append(list(specs)) or [])
    monkeypatch.setattr(script, "default_data_dir", lambda: tmp_path)
    existing = script.result_file_name(next(iter(script.all_runs())), DatasetSplit.TEST)
    (tmp_path / existing).write_text("{}")
    assert script.main(["--split", "test", "--out", str(tmp_path), "--exclude", "mapa:en:on", "--skip-existing"]) == 0
    specs = seen[0]
    assert len(specs) == 54 - 9 - 1
    assert not any(s.dataset == "mapa" and s.language == "en" and s.coreference for s in specs)
    assert all(script.result_file_name(s, DatasetSplit.TEST) != existing for s in specs)
