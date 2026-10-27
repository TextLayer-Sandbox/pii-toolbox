# PII Toolbox

A set of tools for finding personal information (PII) in text, swapping it for placeholders, and swapping it back in afterwards.

We also provide tools for measuring how accurately detectors find PII, as well as their speed and memory use.


## What is in the repository

- `src/pii_toolbox/`: the Python package (shared types, detectors, name linking, masking sessions, datasets and the evaluation code).
- `examples/`: a small runnable example that masks a text and restores it.
- `scripts/`: command-line scripts to download data and models, run evaluations and rebuild the results.
- `tests/`: the automated tests.
- `results/`: saved scores, tables and charts from the evaluation runs.
- `docs/`: the longer guides listed below.

## Quick start

```
uv sync
uv run pytest -q
uv run python examples/round_trip.py
```

## Further reading

- [Masking toolbox](docs/masking-toolbox.md): how text is masked and restored, with a walk-through of the example.
- [Evaluation](docs/evaluation.md): the datasets, how detectors are scored, and how to rerun an evaluation.
- [Results](docs/results.md): what each saved table and chart shows and how they are rebuilt.
- [Results notebook](notebooks/results.ipynb): the results explained with charts, and a way to score your own detector on your own data.
- [Development](docs/development.md): setup, tests and the layout of the repository's working files.
