"""Reads and writes evaluation result files: aggregate scores and the run environment, never document content."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ResultRecord:
    """A saved run with the facts about its detector that the tables need.

    `languages` is the detector's supported ISO codes, None meaning any language;
    `detects` is the set of types the detector can report; `language` is the single
    language the run was measured on, None when the file covers every language of its dataset.
    """

    run: EvaluationRun
    languages: frozenset[str] | None
    detects: frozenset[PiiType]
    language: str | None = None


def run_file_name(tool: str, dataset: str, split: DatasetSplit, coreference: bool) -> str:
    return f"{tool}__{dataset}__{split.name.lower()}__coref-{'on' if coreference else 'off'}.json"


def language_from_file_name(file_name: str) -> str | None:
    """The language segment of `<tool>__<dataset>__<language>__<split>__coref-<state>.json`, None for a name without one."""
    parts = file_name.split("__")
    return parts[2] if len(parts) == 5 else None


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


def write_run(record: ResultRecord, out_dir: Path) -> Path:
    """Writes the result file and the environment file into out_dir; returns the result file path."""
    run = record.run
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / run_file_name(run.tool, run.dataset, run.split, run.coreference)
    content = run_to_dict(run)
    content["detector_languages"] = None if record.languages is None else sorted(record.languages)
    content["detector_detects"] = sorted(pii_type.name for pii_type in record.detects)
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    (out_dir / ENVIRONMENT_FILE_NAME).write_text(
        json.dumps(_environment_to_dict(run.environment), indent=2) + "\n", encoding="utf-8"
    )
    return path


def read_record(path: Path) -> ResultRecord:
    raw = json.loads(path.read_text(encoding="utf-8"))
    languages = raw["detector_languages"]
    return ResultRecord(
        run=run_from_dict(raw),
        languages=None if languages is None else frozenset(languages),
        detects=frozenset(PiiType[name] for name in raw["detector_detects"]),
        language=language_from_file_name(path.name),
    )


def read_run(path: Path) -> EvaluationRun:
    return read_record(path).run


def load_for_selection(path: Path) -> EvaluationRun:
    """Reads a result file to base a choice on; refuses a file measured on the test split."""
    run = read_run(path)
    if run.split is DatasetSplit.TEST:
        raise SelectionFromTestSplitError(f"{path.name} was measured on the test split; choices use dev or validation only")
    return run
