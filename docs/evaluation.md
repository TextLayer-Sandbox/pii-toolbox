# Evaluation

This guide explains the datasets the detectors are measured on, how a run is scored, and how to rerun an evaluation. The code lives in `src/pii_toolbox/datasets` and `src/pii_toolbox/evaluation`; the command-line scripts are in `scripts/`. What the saved results look like is covered in [Results](results.md).

## Datasets

All three are public. They are downloaded into a data folder outside the repository, set with the `PII_DATA_DIR` environment variable (default `~/.cache/pii-toolbox/data`). Each download is pinned to a fixed commit so everyone gets the same files.

### TAB (Text Anonymization Benchmark)

English court judgments from the European Court of Human Rights, annotated by several people. Each mention is marked as a direct identifier (identifies a person on its own), a quasi identifier (identifies someone in combination with other details) or not needing masking. Mentions also carry an entity id that ties together every mention of one person or thing. TAB has the types person, organisation, location, date and time, quantity and code (which maps to the common `IDENTIFIER` type). Its `DEM` and `MISC` labels have no common type, so they are scored separately from the common types. The loader is `datasets/tab.py`.

### MAPA

A multilingual corpus covering 21 languages. Each file is split into sentences with word-by-word tags; the loader joins the words and rebuilds character positions for each annotated span. The types are person, organisation, address (mapped to location), date and time (merged into one type) and amount (mapped to quantity). MAPA has no risk grading and no entity ids. The evaluation runs it for English, French and German. Its "dev" split is MAPA's own validation file. The loader is `datasets/mapa.py`.

### Worldwide English NER

English newswire articles from around the world, from the Stanford `en-worldwide-newswire` repository. Its types are person, organisation, location, date and money (mapped to quantity); facility, NORP, miscellaneous and product have no common type and are left out of the counts. Every article is used whatever split is asked for, because nothing is tuned on them. The loader is `datasets/worldwide.py`.

### Structured PII datasets

Two public synthetic datasets from Hugging Face, used only for the per-kind measurement of Presidio's recognisers on IDs, bank numbers and contact details (`scripts/run_structured_evaluation.py`): `nvidia/Nemotron-PII` and `gretelai/synthetic_pii_finance_multilingual`, each pinned to a commit and read from its test split, downloaded into `structured/` under the data folder on first use. The loader is `datasets/structured_pii.py`.

### Common types and the answer key

Every tool and dataset is translated onto six common types (`PERSON`, `ORGANIZATION`, `LOCATION`, `DATE_TIME`, `QUANTITY`, `IDENTIFIER`) using the tables in `core/label_maps.py`.

Each dataset lists the common types it annotates: TAB annotates all six; MAPA and Worldwide annotate all except `IDENTIFIER`. A type outside that list can never be correct for that dataset, so predictions of it are not counted as false alarms (see "Which types are counted" below).

For TAB, several annotators marked each document. The answer key (`datasets/answer_key.py`) counts a span as PII if any annotator marked it as direct or quasi. Identical spans from several annotators count once; spans with different boundaries stay separate. Datasets without risk grading keep every span.

## How a run is scored

A run takes one detector setup, one dataset and one split, finds spans in every document, and compares them with the answer key. Only totals are saved, never document text.

### Detector setups

A setup is a detector name from the registry, or a union written with `+`. The evaluation plan (`evaluation/configs.py`) measures:

- `presidio_structured` alone,
- each model detector alone: `gliner`, `roberta`, `spacy_lg`, `spacy_sm`,
- each model detector joined with `presidio_structured`, such as `gliner+presidio_structured`.

That is nine setups. There are no unions of two models. The `known_names` detector is not part of the plan.

For each setup the plan runs: TAB, MAPA (English) and Worldwide (English), each with co-reference off and on; and MAPA French and German with co-reference off only, because the co-reference model reads English only. "Co-reference on" means names the detector missed are added when another form of the same person was found (see [Masking toolbox](masking-toolbox.md)).

A detector that does not support a document's language skips that document. This is why `roberta` and the spaCy setups have no French or German scores.

### Matching: strict and relaxed

Every run is scored two ways:

- **Strict** (`EXACT_BOUNDARIES`): a prediction matches an answer only if it has the same start and end and the same common type.
- **Relaxed** (`ANY_OVERLAP`): a prediction matches an answer of the same type if the overlap covers at least half of the longer of the two spans. A fragment that clips a small part of an answer, or a very long prediction around a short answer, does not match.

In both cases, matching is one to one: each prediction is used at most once, and the pairs with the longest overlap are taken first. Prediction and answer must have the same common type. Answers with no common type (such as TAB's `DEM` and `MISC`) take no part in the normal counts, so a prediction that only overlaps one of those counts as a false positive.

Answers of the same type that overlap each other, directly or through a chain of overlaps, form a group that counts as one entity. A prediction gets credit if it matches any span in the group under the rule in use, and a group is credited at most once. The spans are not merged into one wider span, so strict matching still needs the exact boundaries of one member of the group.

The gap between the two scores shows how often a detector finds the right thing but cuts it short or runs it long.

### Which types are counted

- Predictions of a type the dataset does not annotate are dropped before matching, so they are neither true nor false positives. For MAPA and Worldwide this applies to `IDENTIFIER`.
- Answers of a type the setup cannot report are left out of the overall and by-language scores, so a setup is charged for misses only on the types it looks for. For example, `roberta` reports three types, so on TAB it is scored on those three alone.

Setups are therefore graded on different sets of types. The numbers show how good each setup is at what it attempts, not which one covers more: a setup that looks for fewer types is not penalised for the types it leaves out.

**Reading TAB overall scores for combinations.** On TAB, `roberta`, `spacy_lg` and `spacy_sm` cannot report `IDENTIFIER`, so their overall scores leave out TAB's `IDENTIFIER` answers. Adding `presidio_structured` to any of them (for example `roberta+presidio_structured`) lets the combination report `IDENTIFIER`, so those answers are counted and the combination is scored against a larger set of answers. A drop in overall recall from a single detector to its combination (such as `roberta` against `roberta+presidio_structured`) is therefore not a like-for-like comparison. The same applies to every TAB figure built from the overall score: Table 1 (TAB columns), the overall F2 column of Table 3, the TAB reference column of Table 5, and charts A, C, D and F. The by-type scores (Table 2, chart B), TAB's own ER_di, ER_qi and WP columns in Tables 3 and 4, and the before/after comparison within one setup in Table 4 are not affected. MAPA and Worldwide have no `IDENTIFIER` answers and are not affected.

### Precision, recall and F2

- **Precision**: the share of predictions that matched an answer.
- **Recall**: the share of answers that were matched by a prediction.
- **F2**: a single score that counts recall twice as heavily as precision, because a missed piece of PII is worse than an over-masked word. It is `5 x precision x recall / (4 x precision + recall)`.

A report holds these scores in several cuts:

- **Overall (micro)**: counts pooled across the types the setup can report, so the larger types weigh more.
- **Macro**: a plain average of the per-type scores, over the types that the detector can report and that appear in the data.
- **By type**: one score per common type. A type the detector cannot report, or one with no answers and no predictions in the data, is shown as `n/a`.
- **By language**: pooled score for each document language, over the same types as the overall score.

`IDENTIFIER` is scored on TAB only: MAPA and Worldwide do not annotate it, so it is left out of their per-type tables.

### TAB's own metrics

For TAB, the run also computes the benchmark's published measures (`evaluation/tab_metrics.py`):

- **ER_di**: entity recall on direct identifiers.
- **ER_qi**: entity recall on quasi identifiers.
- **WP**: weighted precision over word tokens, where each token is weighted by how much information it carries.

An entity counts as found only when every mention that needs masking is covered. These metrics need the English spaCy medium model and `bert-base-uncased`, both fetched by the model download.

### Cost

Cost is measured separately from accuracy, with one fresh process per setup so memory belongs to that setup alone. It records peak memory, model load time, throughput at 1,000, 10,000 and 100,000 characters, median time per 1,000 characters, and whether the setup fits in 10 GB (the size of an AWS Lambda function).

## Rerunning an evaluation

Everything below is run from the repository root. Install first with `uv sync --group models`: the downloads and the table and chart rebuild need only the default groups, but running the model-based detectors, co-reference and TAB's weighted precision needs the `models` group (GLiNER, transformers and torch).

### 1. Download datasets and models

```
uv run python scripts/download_datasets.py
uv run python scripts/download_models.py
```

Both skip files already present. Each takes `--dest <folder>` to choose where files go. Without it, datasets go to `PII_DATA_DIR` and models to `PII_MODEL_DIR` (defaults `~/.cache/pii-toolbox/data` and `~/.cache/pii-toolbox/models`). The model download is large: it includes GLiNER, RoBERTa large, the spaCy models, the Maverick co-reference model, and the encoders used for TAB's weighted precision.

### 2. Run one setup on one dataset

```
uv run python -m pii_toolbox.evaluation run \
  --tool gliner+presidio_structured --dataset tab --split test \
  --coref off --out one-run-output
```

Options: `--tool` is a detector name or a `+` union; `--dataset` is `tab`, `mapa` or `worldwide`; `--split` is `train`, `dev`, `validation` or `test` (`validation` is the same as `dev`); `--coref` is `off` or `on`; `--language` restricts the run to one ISO code (for MAPA); `--data-dir` overrides the data folder. The result file and an `environment.json` (machine, operating system and library versions) are written to `--out`.

### 3. Run the whole plan

```
uv run python scripts/run_all_evaluations.py --split test --out results/test
```

This runs every setup and dataset in the plan and writes one file per run, named `<setup>__<dataset>__<language>__<split>__coref-<off|on>.json`, plus one `environment.json`. A run that fails is reported at the end and the exit code is 1; the rest still run. Useful options:

- `--coref off` skips the co-reference runs.
- `--only-config spacy_sm` runs one setup.
- `--exclude mapa:fr:off` leaves out one kind of run (written `dataset:language:on|off`; can be repeated).
- `--skip-existing` leaves alone any run whose file already exists.
- `--max-documents 5` scores only the first five documents of each load, for a quick trial.
- `--split dev` scores the development split instead of test.
- `--data-dir` overrides the data folder.

A full run loads large models and takes a long time. The published figures use the test split only.

### 4. Measure cost

```
uv run python scripts/run_cost_all.py
```

This measures each of the nine setups in its own process and writes `results/cost/<setup>.json` (with `+` in the name written as `__`). It takes no options, loads every model, and overwrites the files in `results/cost/`.

### 5. Rebuild the tables and charts

```
uv run python scripts/build_outputs.py --results results/test --cost results/cost --out results/outputs
```

This reads only the saved files: no dataset is read and no model is loaded. See [Results](results.md).
