"""Reads and writes evaluation result files: aggregate scores and the run environment, never document content."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pii_toolbox.core import (
    DatasetSplit,
    DetectionReport,
    DetectionScore,
    EvaluationRun,
    PiiType,
    RunEnvironment,
    SpanMatchRule,
    TabOfficialMetrics,
)

ENVIRONMENT_FILE_NAME = "environment.json"


class SelectionFromTestSplitError(ValueError):
    """Raised when a choice would be made from a result measured on the test split."""


def run_file_name(tool: str, dataset: str, split: DatasetSplit, coreference: bool) -> str:
    return f"{tool}__{dataset}__{split.name.lower()}__coref-{'on' if coreference else 'off'}.json"


def _score_to_dict(score: DetectionScore) -> dict[str, float]:
    return {"precision": score.precision, "recall": score.recall, "f2": score.f2}


def _report_to_dict(report: DetectionReport) -> dict[str, Any]:
    return {
        "rule": report.rule.name,
        "overall_micro": _score_to_dict(report.overall_micro),
        "macro": _score_to_dict(report.macro),
        "by_type": {
            pii_type.name: None if score is None else _score_to_dict(score) for pii_type, score in report.by_type.items()
        },
        "by_language": {language: _score_to_dict(score) for language, score in report.by_language.items()},
    }


def _environment_to_dict(environment: RunEnvironment) -> dict[str, Any]:
    return {
        "machine": environment.machine,
        "os": environment.os,
        "runtime": environment.runtime,
        "library_versions": dict(environment.library_versions),
    }


def run_to_dict(run: EvaluationRun) -> dict[str, Any]:
    tab = run.tab_metrics
    return {
        "tool": run.tool,
        "coreference": run.coreference,
        "dataset": run.dataset,
        "split": run.split.name,
        "exact_boundaries": _report_to_dict(run.exact_boundaries),
        "any_overlap": _report_to_dict(run.any_overlap),
        "tab_metrics": None if tab is None else {"er_di": tab.er_di, "er_qi": tab.er_qi, "wp": tab.wp},
        "environment": _environment_to_dict(run.environment),
    }


def _score_from_dict(raw: Mapping[str, float]) -> DetectionScore:
    return DetectionScore(raw["precision"], raw["recall"], raw["f2"])


def _report_from_dict(raw: Mapping[str, Any]) -> DetectionReport:
    return DetectionReport(
        rule=SpanMatchRule[raw["rule"]],
        overall_micro=_score_from_dict(raw["overall_micro"]),
        macro=_score_from_dict(raw["macro"]),
        by_type={
            PiiType[name]: None if score is None else _score_from_dict(score) for name, score in raw["by_type"].items()
        },
        by_language={language: _score_from_dict(score) for language, score in raw["by_language"].items()},
    )


def run_from_dict(raw: Mapping[str, Any]) -> EvaluationRun:
    tab = raw["tab_metrics"]
    environment = raw["environment"]
    return EvaluationRun(
        tool=raw["tool"],
        coreference=raw["coreference"],
        dataset=raw["dataset"],
        split=DatasetSplit[raw["split"]],
        exact_boundaries=_report_from_dict(raw["exact_boundaries"]),
        any_overlap=_report_from_dict(raw["any_overlap"]),
        tab_metrics=None if tab is None else TabOfficialMetrics(tab["er_di"], tab["er_qi"], tab["wp"]),
        environment=RunEnvironment(
            environment["machine"], environment["os"], environment["runtime"], environment["library_versions"]
        ),
    )


def write_run(run: EvaluationRun, out_dir: Path) -> Path:
    """Writes the run's result file and the environment file into out_dir; returns the result file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / run_file_name(run.tool, run.dataset, run.split, run.coreference)
    path.write_text(json.dumps(run_to_dict(run), indent=2) + "\n", encoding="utf-8")
    (out_dir / ENVIRONMENT_FILE_NAME).write_text(
        json.dumps(_environment_to_dict(run.environment), indent=2) + "\n", encoding="utf-8"
    )
    return path


def read_run(path: Path) -> EvaluationRun:
    return run_from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_for_selection(path: Path) -> EvaluationRun:
    """Reads a result file to base a choice on; refuses a file measured on the test split."""
    run = read_run(path)
    if run.split is DatasetSplit.TEST:
        raise SelectionFromTestSplitError(f"{path.name} was measured on the test split; choices use dev or validation only")
    return run
