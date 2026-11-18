# Development

How to set the project up, run its tests, and find your way around the working files.

## Setup

The project needs Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

The package has no required dependencies of its own. The optional groups in `pyproject.toml` are `dev` (pytest), `detection` (Presidio and spaCy), `models` (GLiNER, transformers and torch), `evaluation` (dataset loading, plotting, memory measurement and what TAB's scoring needs) and `coref` (ZeroMQ). A plain `uv sync` installs `dev`, `detection`, `evaluation` and `coref`, which is enough to run the tests, the table and chart rebuild, the structured-identifier detectors and the examples that use no model. The `models` and `notebook` groups are left out because they are large. Add them when you need them:

```
uv sync --group models      # GLiNER, RoBERTa, co-reference and TAB's weighted precision
uv sync --group notebook    # opens and runs notebooks/results.ipynb
uv sync --all-groups        # everything
```

`--group` adds to the default groups. Commands are run with `uv run ...` from the repository root.

The co-reference worker (`src/pii_toolbox/coref/worker`) has its own `pyproject.toml` and lock file, because the Maverick model needs older library versions than the rest of the toolbox. `MaverickResolver` starts it with `uv run --project`, so uv builds that environment the first time it is used.

Datasets and models are not part of the repository. See [Evaluation](evaluation.md) for how to download them.

## Running the tests

```
uv run pytest -q
```

This runs the fast tests, which need no downloaded data or models and no network. At the time of writing that is 522 tests passing in a few seconds, with 17 not selected.

### Tests that are not selected by default

`pyproject.toml` sets `-m 'not model and not data and not network'`, so three kinds of test are skipped unless asked for. Each carries a marker:

- `model`: needs model files downloaded (for example the real GLiNER, RoBERTa, spaCy and Maverick models).
- `data`: needs a dataset downloaded (for example the real TAB, MAPA and Worldwide files).
- `network`: is allowed to use the network. Every other test has the network blocked.

Run them by choosing the marker:

```
uv run pytest -m model
uv run pytest -m data
uv run pytest -m "model or data or network"
```

An `-m` on the command line replaces the default filter. The 17 deselected tests are the ones marked `model` or `data`. Before running them, download the files (`scripts/download_datasets.py` and `scripts/download_models.py`) and set `PII_DATA_DIR` and `PII_MODEL_DIR` to the folders that hold them, since tests marked `data` or `model` use the folders you set and every other test is pointed at an empty temporary folder.

### Test safety net

`tests/conftest.py` applies two rules to every test automatically:

- Network calls fail with an error unless the test is marked `network`, so no text can leave the machine during a test.
- Home, cache, data and model folders point into a temporary folder, so tests do not touch your real files. Tests marked `data` or `model` keep the folders you set.

### Layout of `tests/`

The folders mirror the packages in `src/pii_toolbox`:

- `tests/core/`: shared types, label tables, the network block, and checks that `pyproject.toml` declares the packages it should.
- `tests/detection/`: each detector, the registry, unions and the segmenter. Most use stand-in models; the ones that load real models are marked `model`.
- `tests/coref/`: grouping of mentions, the nickname list and step, long documents, the worker protocol, and the real Maverick worker (marked `model`).
- `tests/datasets/`: the dataset loaders, the answer key and the download logic. `tests/datasets/fixtures/` holds small TAB-shaped sample files. Tests against the real corpora are marked `data`.
- `tests/evaluation/`: matching, scoring, TAB's metrics, the run command, the run-all script, cost measurement, tables and charts. `tests/evaluation/fixtures/` holds small saved results with the tables and charts expected from them, used to check the rebuild is exact. Some tests check that the files committed under `results/` are complete and consistent.
- `tests/session/`: the masking session, the placeholder format, the store and the example script.

## Other top-level files

- `examples/round_trip.py`: the runnable masking example, explained in [Masking toolbox](masking-toolbox.md).
- `scripts/`: command-line scripts, explained in [Evaluation](evaluation.md).
