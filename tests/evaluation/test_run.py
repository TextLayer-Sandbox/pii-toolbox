import json

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
from pii_toolbox.evaluation.__main__ import main
from pii_toolbox.evaluation.results_io import (
    ENVIRONMENT_FILE_NAME,
    SelectionFromTestSplitError,
    load_for_selection,
    read_run,
    write_run,
)
from pii_toolbox.evaluation.run import RunRequest, run_evaluation

CANARY = "ZXQCANARYTOKEN"
TEXT = f"Anna Testwell met Anna there. {CANARY} stays secret."
FULL_NAME = (0, 13)
SHORT_NAME = (18, 22)


class StubDetector(Detector):
    """Finds the full name only, in English documents."""

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
    def __init__(self, name, data_dir=None):
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


def run_cli(tmp_path, *extra, dataset="tab", split="dev", coref="off"):
    argv = ["run", "--tool", "stub", "--dataset", dataset, "--split", split, "--coref", coref, "--out", str(tmp_path)]
    return main(
        argv + list(extra),
        registry=stub_registry(),
        dataset_factories={"tab": lambda d: StubDataset("TAB"), "mapa": lambda d: StubDataset("MAPA")},
        tab_scorer=StubTabScorer(),
        resolver=StubResolver(),
    )


def walk(node):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def test_run_writes_the_named_result_file_and_environment_file(tmp_path):
    path = run_cli(tmp_path)
    assert path == tmp_path / "stub__tab__dev__coref-off.json"
    environment = json.loads((tmp_path / ENVIRONMENT_FILE_NAME).read_text())
    assert set(environment) == {"machine", "os", "runtime", "library_versions"}
    assert environment["machine"] and environment["os"] and environment["runtime"]
    assert "pytest" in {name.lower() for name in environment["library_versions"]}


def test_result_holds_both_rules_per_type_per_language_and_tab_metrics(tmp_path):
    run = read_run(run_cli(tmp_path))
    assert run.tab_metrics == TabOfficialMetrics(0.5, 0.25, 0.75)
    assert run.exact_boundaries.by_type[PiiType.PERSON].recall == pytest.approx(0.5)
    assert run.any_overlap.by_type[PiiType.PERSON].recall == pytest.approx(0.5)
    assert set(run.exact_boundaries.by_language) == {"en"}


def test_non_tab_runs_have_no_tab_metrics_and_keep_the_dataset_name(tmp_path):
    path = run_cli(tmp_path, dataset="mapa", split="validation")
    assert path.name == "stub__mapa__dev__coref-off.json"
    assert read_run(path).tab_metrics is None


def test_result_files_hold_no_document_text_and_no_span_lists(tmp_path):
    path = run_cli(tmp_path)
    for written in (path, tmp_path / ENVIRONMENT_FILE_NAME):
        content = written.read_text()
        assert CANARY not in content
        assert "Anna" not in content
        tree = json.loads(content)
        assert not [node for node in walk(tree) if isinstance(node, list)]
        assert not [node for node in walk(tree) if isinstance(node, dict) and {"start", "end"} <= set(node)]


def test_coreference_on_adds_the_resolved_name_form_and_scores_english_only(tmp_path):
    off = read_run(run_cli(tmp_path / "off"))
    on_path = run_cli(tmp_path / "on", coref="on")
    assert on_path.name == "stub__tab__dev__coref-on.json"
    on = read_run(on_path)
    assert on.coreference and not off.coreference
    assert off.exact_boundaries.by_type[PiiType.PERSON].recall == pytest.approx(0.5)
    assert on.exact_boundaries.by_type[PiiType.PERSON].recall == pytest.approx(1.0)


def test_run_requires_a_resolver_when_coreference_is_on():
    request = RunRequest("stub", StubDataset("MAPA"), DatasetSplit.DEV, True)
    with pytest.raises(ValueError):
        run_evaluation(request, stub_registry())


def test_selection_loader_reads_dev_results(tmp_path):
    path = run_cli(tmp_path)
    assert load_for_selection(path).split is DatasetSplit.DEV


def test_selection_loader_refuses_test_split_results(tmp_path):
    path = run_cli(tmp_path, split="test")
    assert read_run(path).split is DatasetSplit.TEST
    with pytest.raises(SelectionFromTestSplitError):
        load_for_selection(path)


def test_write_run_round_trips(tmp_path):
    run = read_run(run_cli(tmp_path / "first"))
    again = write_run(run, tmp_path / "second")
    assert read_run(again) == run
