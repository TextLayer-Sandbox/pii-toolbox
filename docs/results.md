# Results

The `results/` folder holds the saved scores from the evaluation runs and everything built from them. For how the runs are scored see [Evaluation](evaluation.md).

## What is saved

```
results/
  test/                  one JSON file per evaluation run, plus environment.json
  cost/                  one JSON file per detector setup with cost measurements
  outputs/               tables (Markdown and CSV), charts (SVG) and manifest.json
  language_model_findings.md
  language_model_findings/
```

- `results/test/` holds the scores on each dataset's test split, one file per setup, dataset, language and co-reference state, named `<setup>__<dataset>__<language>__test__coref-<off|on>.json`. Each file holds only totals (precision, recall and F2 for strict and relaxed matching, overall and by type and language, plus TAB's own metrics for TAB runs) and the machine and library versions. `environment.json` describes the machine and software the runs used.
- `results/cost/` holds the cost measurements: peak memory, load time, throughput at three input sizes, median time per 1,000 characters, and whether the setup fits in 10 GB.
- `results/outputs/` holds everything below, built from the two folders above.

The setups are `presidio_structured`, `gliner`, `roberta`, `spacy_lg` and `spacy_sm`, plus each model with `presidio_structured` (written `gliner+presidio_structured`, and so on).

## Tables

Each table comes as a Markdown file (`.md`) for reading and a CSV file (`.csv`) for other tools. Numbers have three decimals. Rows list single detectors first and then combinations. "Strict" and "relaxed" are the two matching rules explained in [Evaluation](evaluation.md).

| File | What it shows |
| --- | --- |
| `table1` | Overall precision, recall and F2 for every setup on TAB, MAPA (English) and Worldwide, each with strict and relaxed matching. |
| `table2_strict`, `table2_relaxed` | F2 for each setup split by PII type, per dataset. `n/a` means the setup cannot report that type or the data has none; a blank means the type is not scored for that dataset (identifiers are scored on TAB only). |
| `table3` | TAB's own measures (ER_di, ER_qi and WP) next to the overall F2 on TAB. TAB only. |
| `table4_strict`, `table4_relaxed` | English text with co-reference off and on: F2 before, after and the change, and for TAB also entity-level recall on direct and quasi identifiers before, after and the change. MAPA and Worldwide rows have no recall columns, because their data has no entity ids. |
| `table5_strict`, `table5_relaxed` | F2 for MAPA English, French and German (and any other language present), with TAB as an English reference column. `n/a` means the setup does not support that language. |
| `table6` | Cost and speed for each setup: peak memory, load time, throughput at 1,000, 10,000 and 100,000 characters, time per 1,000 characters, and whether it fits in 10 GB. There is one row per setup, as there is no dataset involved. |

## Charts

Charts are SVG files drawn from the same saved results. They use strict matching unless stated, and are drawn the same way every time. Charts A to D and F use TAB; chart E uses MAPA.

| File | What it shows |
| --- | --- |
| `chart_a.svg` | Recall against precision on TAB, one labelled point per setup, with F2 contour lines at 0.5, 0.7 and 0.9. |
| `chart_b.svg` | F2 by PII type on TAB: a group of bars per type, one bar per setup. A type a setup cannot report is drawn at zero. |
| `chart_c.svg` | Strict against relaxed overall F2 on TAB, a pair of bars per setup. The gap is the cases where a detector found the item but got its edges wrong. |
| `chart_d.svg` | Accuracy against cost: TAB F2 against time per 1,000 characters (log scale), with marker area proportional to peak memory. |
| `chart_e.svg` | F2 on MAPA by language, one line per setup. A language a setup was not scored on is a gap in its line. |
| `chart_f.svg` | Overall TAB F2 with co-reference off and on, for each setup. |

## The manifest

`results/outputs/manifest.json` records, for every table cell and every chart, which saved file or files it was computed from. Use it to trace a number back to its source.

## Rebuilding

```
uv run python scripts/build_outputs.py --results results/test --cost results/cost --out results/outputs
```

This reads only the saved files: no dataset is read, no model is loaded, and it takes seconds. It refuses any result file measured on a split other than test and names the file; it also stops if the cost folder does not exist. Running it twice on the same files produces identical output.

To produce new input files rather than rebuild from existing ones, see the rerun steps in [Evaluation](evaluation.md).

## Language-model findings

`results/language_model_findings.md` is a written note on measuring a local general-purpose language model (Qwen3 8B at 4-bit) as a detector. It records the machine, two rounds of timing on the same four TAB development passages (before and after fixing how the model was run), where the time went, and why this approach was judged too slow for bulk redaction. It holds totals only. The raw timings are in `results/language_model_findings/runtime_timing_after_fix.json`. The note is a record of a measurement; the toolbox has no language-model detector.
