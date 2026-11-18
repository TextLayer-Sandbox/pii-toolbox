import importlib.util
import json
from pathlib import Path

import pytest

from pii_toolbox.core import DatasetSplit
from pii_toolbox.evaluation.configs import RunSpec
from pii_toolbox.evaluation.results_io import ENVIRONMENT_FILE_NAME
from stubs import StubDataset, StubResolver, StubTabScorer, stub_registry

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_all_evaluations.py"
_spec = importlib.util.spec_from_file_location("run_all_evaluations", SCRIPT)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

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
    assert not any(s.dataset == "mapa" and s.language == "en" and s.coreference for s in specs)
    assert all(script.result_file_name(s, DatasetSplit.TEST) != existing for s in specs)


def _main_out_dir(monkeypatch, tmp_path, argv):
    seen = []
    monkeypatch.setattr(script, "run_all", lambda split, out, specs, **kw: seen.append(out) or [])
    monkeypatch.setattr(script, "default_data_dir", lambda: tmp_path)
    assert script.main(argv) == 0
    return seen[0]


def test_max_documents_writes_to_a_partial_folder(monkeypatch, tmp_path):
    out = _main_out_dir(monkeypatch, tmp_path, ["--split", "test", "--out", str(tmp_path / "res"), "--max-documents", "2"])
    assert out == tmp_path / "res-partial"


def test_default_out_folder_follows_the_split(monkeypatch, tmp_path):
    assert _main_out_dir(monkeypatch, tmp_path, ["--split", "dev"]) == Path("results/dev")
    assert _main_out_dir(monkeypatch, tmp_path, ["--split", "dev", "--max-documents", "1"]) == Path("results/dev-partial")


def test_unknown_only_config_is_rejected(capsys):
    with pytest.raises(SystemExit) as exit_info:
        script.main(["--split", "test", "--only-config", "no_such_config"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
