import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

from pii_toolbox.evaluation.build_all import MANIFEST_FILE_NAME, build_outputs

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build_outputs.py"
FIXTURES = Path(__file__).parent / "fixtures" / "build_all"
RESULTS = FIXTURES / "results"
GOLDEN = FIXTURES / "golden"
KEY_COLUMNS = {"table1": 1, "table2_strict": 2, "table2_relaxed": 2, "table3": 1, "table4_strict": 2, "table4_relaxed": 2,
               "table5_strict": 1, "table5_relaxed": 1, "table6": 1}


def load_script():
    spec = importlib.util.spec_from_file_location("build_outputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_file(source):
    """A cost source is a repo-root-relative or absolute path; a result source is a name in RESULTS."""
    if "/" in source:
        return REPO_ROOT / source
    return RESULTS / source


def contents(folder):
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


def test_two_builds_are_byte_identical_and_equal_the_golden_files(tmp_path):
    build_outputs(RESULTS, tmp_path / "one")
    build_outputs(RESULTS, tmp_path / "two")
    first, second = contents(tmp_path / "one"), contents(tmp_path / "two")
    assert first == second == contents(GOLDEN)


def test_every_table_and_chart_is_written(tmp_path):
    names = {path.name for path in build_outputs(RESULTS, tmp_path)}
    assert {f"chart_{letter}.svg" for letter in "abcdef"} <= names
    for table in KEY_COLUMNS:
        assert {f"{table}.md", f"{table}.csv"} <= names
    assert MANIFEST_FILE_NAME in names


def test_span_score_tables_show_strict_and_relaxed(tmp_path):
    build_outputs(RESULTS, tmp_path)
    header1 = (tmp_path / "table1.csv").read_text().splitlines()[0]
    assert "TAB F2 strict" in header1 and "MAPA (en) F2 relaxed" in header1
    header3 = (tmp_path / "table3.csv").read_text().splitlines()[0]
    assert "Any-annotator F2 strict" in header3 and "Any-annotator F2 relaxed" in header3
    for table in ("table2", "table4", "table5"):
        strict, relaxed = (tmp_path / f"{table}_strict.md").read_text(), (tmp_path / f"{table}_relaxed.md").read_text()
        assert "strict matching" in strict and "relaxed matching" in relaxed
        assert strict != relaxed


def test_manifest_maps_every_filled_cell_and_chart_to_existing_source_files(tmp_path):
    build_outputs(RESULTS, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE_NAME).read_text())
    for table, key_columns in KEY_COLUMNS.items():
        with open(tmp_path / f"{table}.csv", newline="") as handle:
            rows = list(csv.reader(handle))
        filled = sum(1 for row in rows[1:] for cell in row[key_columns:] if cell != "")
        assert len(manifest["tables"][table]) == filled
        for sources in manifest["tables"][table].values():
            assert sources and all(source_file(source).is_file() for source in sources)
    assert set(manifest["charts"]) == {f"chart_{letter}.svg" for letter in "abcdef"}
    for sources in manifest["charts"].values():
        assert sources and all(source_file(source).is_file() for source in sources)


def test_a_cell_is_traced_to_the_file_it_came_from(tmp_path):
    build_outputs(RESULTS, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE_NAME).read_text())
    assert manifest["tables"]["table3"]["spacy_lg | ER_di"] == ["spacy_lg__tab__test__coref-off.json"]
    assert manifest["tables"]["table4_strict"]["TAB / spacy_lg | ER_di on"] == ["spacy_lg__tab__test__coref-on.json"]
    assert manifest["tables"]["table6"]["gliner | Peak memory (GiB)"] == [(RESULTS / "cost" / "gliner.json").relative_to(REPO_ROOT).as_posix()]


def add_dev_result(results_copy):
    source = results_copy / "spacy_lg__tab__test__coref-off.json"
    content = json.loads(source.read_text())
    content["split"] = "DEV"
    (results_copy / "spacy_lg__tab__dev__coref-off.json").write_text(json.dumps(content))


def test_a_result_file_from_another_split_is_refused_and_named(tmp_path, capsys):
    copy = tmp_path / "results"
    shutil.copytree(RESULTS, copy)
    add_dev_result(copy)
    code = load_script().main(["--results", str(copy), "--out", str(tmp_path / "out")])
    assert code != 0
    assert "spacy_lg__tab__dev__coref-off.json" in capsys.readouterr().err
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_the_command_succeeds_with_no_data_or_model_directory_and_loads_no_model(tmp_path):
    program = (
        "import sys, runpy\n"
        "sys.argv = ['build_outputs.py', '--results', sys.argv[1], '--out', sys.argv[2]]\n"
        "try:\n"
        "    runpy.run_path(%r, run_name='__main__')\n"
        "except SystemExit as done:\n"
        "    code = done.code\n"
        "heavy = [name for name in ('torch', 'transformers', 'spacy', 'datasets') if name in sys.modules]\n"
        "print('HEAVY', heavy)\n"
        "sys.exit(code)\n"
    ) % str(SCRIPT)
    environment = {"PATH": "/usr/bin:/bin", "PII_DATA_DIR": str(tmp_path / "absent"), "PII_MODEL_DIR": str(tmp_path / "absent")}
    result = subprocess.run(
        [sys.executable, "-c", program, str(RESULTS), str(tmp_path / "out")],
        capture_output=True, text=True, env=environment,
    )
    assert result.returncode == 0, result.stderr
    assert "HEAVY []" in result.stdout
    assert (tmp_path / "out" / "table1.md").is_file()


def test_cost_folder_outside_the_results_folder_is_named_with_the_cost_option(tmp_path, capsys):
    results = tmp_path / "results"
    shutil.copytree(RESULTS, results)
    outside_cost = tmp_path / "elsewhere" / "cost_files"
    outside_cost.parent.mkdir()
    (results / "cost").rename(outside_cost)
    out = tmp_path / "out"
    code = load_script().main(["--results", str(results), "--cost", str(outside_cost), "--out", str(out)])
    assert code == 0, capsys.readouterr().err
    assert (out / "table6.md").read_text() == (GOLDEN / "table6.md").read_text()
    manifest = json.loads((out / MANIFEST_FILE_NAME).read_text())
    expected = str((outside_cost / "gliner.json").resolve())
    assert manifest["tables"]["table6"]["gliner | Peak memory (GiB)"] == [expected]
    assert Path(expected).is_file()


def test_a_missing_cost_folder_is_reported_when_named_with_the_cost_option(tmp_path, capsys):
    code = load_script().main(["--results", str(RESULTS), "--cost", str(tmp_path / "absent"), "--out", str(tmp_path / "out")])
    assert code != 0
    assert "absent" in capsys.readouterr().err


def test_every_filled_worldwide_cell_maps_to_a_worldwide_result_file(tmp_path):
    build_outputs(RESULTS, tmp_path)
    manifest = json.loads((tmp_path / MANIFEST_FILE_NAME).read_text())
    worldwide_cells = {
        (table, cell): sources
        for table, cells in manifest["tables"].items()
        for cell, sources in cells.items()
        if "Worldwide" in cell
    }
    assert {table for table, _ in worldwide_cells} == {"table1", "table2_strict", "table2_relaxed", "table4_strict", "table4_relaxed"}
    for sources in worldwide_cells.values():
        assert all("__worldwide__en__test__" in source for source in sources)
    assert manifest["tables"]["table1"]["gliner | Worldwide F2 strict"] == ["gliner__worldwide__en__test__coref-off.json"]
    assert manifest["tables"]["table4_strict"]["Worldwide / gliner | F2 on"] == ["gliner__worldwide__en__test__coref-on.json"]
