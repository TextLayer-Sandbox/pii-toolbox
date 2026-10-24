# PII redaction toolbox: plan

## Purpose

The toolbox is a public GitHub repository of local PII detection and evaluation tools, along with a session mechanism that swaps PII out of text and back in. It exists so that we can show, with measurements, what redaction catches, what it misses and what it costs to run, and so that the article "A CTO's guide to PII redaction" rests on numbers a reader can reproduce. PII, short for personally identifiable information, is any data that can be tied to a real person.

No text should be sent to a hosted model, because the point of redaction is that PII never leaves our own machines. Every tool must run comfortably on the reference machine: Apple M5, 32 GB memory, 10 cores.
In practice, we'd expect much more conservative resource limitations, for example, an AWS Lambda execution environment.

## Reference Implementation

Take what you can from this reference implementation to guide you: /Users/liam/tl/poc/poc-anonymizer.

## The tools

Detection tools are separate, swappable detectors behind one common interface (text in, list of typed character spans out). Each is evaluated on its own and in combination.

Identity resolution is another module with sub-tools that can be swapped in or out.


### Presidio, the backbone

Presidio is Microsoft's open-source PII toolkit. We use two things from it: its pattern and checksum recognisers for structured PII (emails, payment cards, IBANs, phone numbers, national and tax IDs, IP addresses and the like), and its interface for plugging in other recognisers. A checksum recogniser checks a candidate against the number's own validation rule, for example the check digit on a card number, so a random digit string is not flagged.

We say plainly what it is: a collection of regular expressions and a convenient interface, not a complete solution. It finds well-formed structured values. It does not understand text, and the repository's results will show where that stops.

### Small trained models for names, places and organisations

These are the models the spike used. Each is run as a Presidio recogniser, so they are swapped by configuration.

| Tool | What it detects | Why it is included |
|---|---|---|
| GLiNER, `urchade/gliner_multi_pii-v1` | Person, organisation, location, address, and other labels you name at run time | Multilingual, and a "zero-shot" model: it takes the labels as a prompt rather than being trained on a fixed list |
| RoBERTa, `Jean-Baptiste/roberta-large-ner-english` | Person, location, organisation (and a catch-all MISC label we ignore) | A large, well-known English entity tagger; a mid-cost reference point |
| spaCy `en_core_web_lg` | Person, location, organisation, dates, quantities and more | The standard, fast statistical pipeline; the large model |
| spaCy `en_core_web_sm` | Same labels | The smallest, cheapest model; the floor for cost |

### A local general language model as a detector

A general-purpose language model, run locally, is prompted to list the PII in a passage and return it as structured output (the exact text of each item and its type). We map each returned item back to character positions in the original text. Because model output can be wrong about quotes and whitespace, matching back to the text is part of the tool and is itself tested.

The model is likely to be quantised, meaning its weights are stored at reduced precision (for example 4-bit) so it fits and runs fast on a laptop-class machine.

The choice of model is to be made. Selection criteria:

- Runs comfortably on the reference machine, with memory to spare for the other tools loaded alongside it.
- Strong in several languages, because MAPA is scored per language.
- Follows an output format reliably (valid structured output on nearly every call).
- Runs through a local runtime (llama.cpp or MLX) with no network access.
- Licence permits redistribution of our results and use in a public repository.

Plausible candidates, to be shortlisted and tried on the TAB and MAPA development splits before we fix one: a Qwen3 model around 8 billion parameters, a Gemma 3 model around 12 billion, a Mistral Nemo 12-billion model, and a Llama 3.1 8-billion model, each at 4-bit quantisation. The final pick, and its exact quantisation, will be recorded in the repo and in the article.

### Co-reference resolution

Co-reference resolution links the different ways a text refers to one entity. Here it is used for people's names only. It links "David Parker", "Dave" and "Mr. Parker" so they share one placeholder, and it masks a form the detector missed when another form of the same person was caught. It never invents a person: a link is only used when the group already contains a detected name.

It is its own tool, a Maverick-based step (Maverick is a co-reference model) that wraps the output of any detector. It runs as a separate worker process because its dependencies do not coexist with the others. It returns only character positions, never text. We measure every detector with it switched on and off.

### How tools are combined

Presidio's structured recognisers always run, since they cover a different kind of PII than the models do. For the name, place and organisation models, the combinations we measure are:

1. Each tool alone (with Presidio's structured recognisers, so structured PII is always covered).
2. Union: a span counts as masked if any tool in the set found it. This follows the "recall first" logic: a miss is a leak.
3. Union of every pair of the model tools, and of all of them together.
4. Each of the above with co-reference on and off.

Overlapping spans from different tools are merged into the widest span before masking. Intersection or voting is not a goal; it would trade recall for precision, the wrong direction here.

## Swapping PII out and back in

The session design is carried over from the spike. It is a design, not a copy of the code.

- **One session per case.** A session is opened when a case (a document, ticket or conversation) arrives and closed when the work on it ends.
- **Placeholders.** Each detected value is replaced by a typed, numbered placeholder such as `<PERSON_1>` or `<EMAIL_ADDRESS_1>`. The number is the count of that type within the session.
- **Stable within a session.** The same text always gets the same placeholder, so the model reading the redacted text can still tell that two mentions are the same person. This holds across everything handled in the session, including later text such as tool results that come back and are masked in turn. With co-reference on, different forms of one person's name also share one placeholder, and the stored original is the fullest form.
- **Mapping in memory only.** The table from placeholder to original value lives only in the memory of the redaction process, keyed by session. It is never written to disk or to logs. Logs record placeholders and counts, never originals.
- **Disposed at the end.** When the session ends, normally or on error, the mapping is deleted.
- **Restoring.** To restore, the original values are substituted for the placeholders in the model's output. Matching tolerates a model lightly mangling a placeholder (dropped brackets, changed case, so `<PERSON_1>`, `PERSON_1` and `person_1` all resolve), but only swaps tokens that exist in this session's mapping. Look-alike strings that were never minted, such as internal handles, pass through untouched. Presidio's own reversal engine needs exact character offsets, which free-form model output does not have, so restoring is plain text substitution.
- **One boundary.** Redacted text goes out; placeholders in what comes back are restored at a single point. Anything returned by a backend system is masked again before the model sees it. Anything past that point is real data, anything before it is masked.
- **Leftover placeholders** that did not resolve are reported, because they point to a model mangling or inventing a token.

The repo includes a runnable example of the full round trip with a stand-in for the model (a function that echoes and edits the text), so a reader sees the mechanism without any model involved.

## Datasets

Two public datasets. Both are downloaded by a script at run time; the repo contains no copy of either.

### TAB (Text Anonymization Benchmark)

TAB is from Pilán et al. 2022 (https://arxiv.org/abs/2202.00443; code and data at https://github.com/NorskRegnesentral/text-anonymization-benchmark). MIT licence. It is 1,268 English judgments of the European Court of Human Rights, split into 1,014 training, 127 development and 127 test documents.

- Entity types: PERSON, CODE, LOC, ORG, DEM, DATETIME, QUANTITY, MISC.
- Each mention is labelled DIRECT (identifies a person on its own), QUASI (identifies in combination with other details) or NO_MASK (needs no masking), with a confidential status and a co-reference id that ties mentions of one entity together.
- Several annotators mark each document (about 4.3 on development and test), and their annotations are kept separate.

How we use it:

- Report TAB's official metrics as they are: entity-level recall on direct identifiers (ER_di) and on quasi-identifiers (ER_qi), where an entity counts as found only if all of its mentions are masked, and token-level weighted precision (WP), which weights each token by an information-content score from a BERT model, micro-averaged over annotators.
- Also compute F2 against an "any annotator" answer key: a span is PII if at least one annotator marked it DIRECT or QUASI. Where annotators mark the same text with different boundaries, the answer key keeps each distinct span, and matching is one-to-one, so a predicted span is credited to at most one answer-key span. We also report recall separately for DIRECT and QUASI spans.
- Because TAB carries co-reference ids, it is where the co-reference tool is measured most closely, with entity-level recall alongside span-level F2.

### MAPA

MAPA is on Hugging Face (https://huggingface.co/datasets/joelniklaus/mapa). CC BY 4.0. It holds EU legal texts from EUR-Lex in 21 languages, including French and German, with sentence-level BIO labels (a tag per word marking whether it begins, continues or is outside an entity). The coarse types are ORGANISATION, ADDRESS, DATE, PERSON, AMOUNT and TIME, with 22 fine types underneath. A single annotator labelled it. It has no ID class and no direct or quasi labels.

How we use it: label spans are rebuilt from the BIO tags and scored on the shared types below. Language is an evaluation dimension: every result is broken out by language, with English, French and German as the headline languages and the full list in the repo's tables.

### Common label set

Both datasets are mapped to one set of PII types so that tools are scored the same way on each. Tool outputs (Presidio entity names, GLiNER labels, language model types) are mapped to the same set.

| Common type | TAB label | MAPA coarse label | Tool outputs mapped to it |
|---|---|---|---|
| PERSON | PERSON | PERSON | PERSON, GLiNER person and name, RoBERTa PER |
| ORGANIZATION | ORG | ORGANISATION | ORGANIZATION, GLiNER organization, RoBERTa ORG |
| LOCATION | LOC | ADDRESS | LOCATION, GLiNER location and address, RoBERTa LOC |
| DATE_TIME | DATETIME | DATE and TIME (merged) | DATE_TIME |
| QUANTITY | QUANTITY | AMOUNT | any tool that emits a quantity or amount type |
| IDENTIFIER | CODE | none | Presidio's structured recognisers (email, phone, card, IBAN, IP, URL, national and other IDs) |

Decisions in this mapping:

- MAPA's DATE and TIME are merged because TAB has a single DATETIME type.
- MAPA's ADDRESS is matched with TAB's LOC: both cover places and addresses.
- TAB's DEM and MISC have no counterpart in MAPA or in the tools; they are reported in the TAB tables as extra rows and left out of the shared-type score.
- IDENTIFIER is scored on TAB only, since MAPA has no such class.
- The "overall" score covers the shared types (PERSON, ORGANIZATION, LOCATION, DATE_TIME, QUANTITY) on both datasets. The TAB tables also show an overall that includes IDENTIFIER.

Choices are fixed on the development split (TAB dev, MAPA validation) and every reported number comes from the test split.

## Evaluation

All of it is code in the repo; nothing is computed by hand.

### Metrics

- **Headline: F2.** F2 is the F-score with beta = 2, which weights recall four times as heavily as precision in the underlying formula. We use it because the two errors are not equal: a miss is a leak (real PII goes onward), while a false alarm only over-masks a harmless word. Reported per PII type and overall, with precision and recall alongside.
- **Span level.** The unit scored is a span (a start and end position in the text) with a type. A predicted span is a true positive if it matches an answer-key span of the same type. Matching is greedy and one-to-one: each answer-key span is matched by at most one prediction, and an unmatched prediction is a false positive, an unmatched answer-key span a false negative.
- **Strict and relaxed matching.** Strict needs exact start and end. Relaxed needs only that the two spans overlap. Every table shows both: the gap between them shows how often a tool finds the right thing but cuts it wrongly, which can leave part of a name unmasked.
- **Overall** pools the counts of all types (micro). We also give the plain average of per-type scores (macro) so rare types are not hidden.
- **TAB official metrics** (ER_di, ER_qi, WP) are computed with TAB's own definitions, reported alongside F2.

### Cost and speed

Measured per tool, and per combination, on the reference machine, with each tool run in its own fresh process so memory figures belong to that tool alone:

- **Memory:** peak resident memory after loading and while processing.
- **Model load time:** seconds from process start to ready.
- **Throughput:** characters per second, over several input sizes (about 1,000, 10,000 and 100,000 characters), since fixed per-call overhead dominates short inputs.
- **Latency per 1,000 characters:** median over repeated runs on warmed models.

The machine, operating system, runtime and library versions are recorded in an environment file written by the run, so figures are tied to what produced them.

### Comparisons

1. Each tool alone.
2. Combinations: unions as described under "How tools are combined".
3. Co-reference off versus on, for each detector and for the best combination.
4. By PII type.
5. By language (MAPA), with TAB as the English reference.
6. Strict versus relaxed matching throughout.
7. Accuracy against cost: F2 for each tool or combination placed next to its memory and latency.

What a reader can rerun: install, run the dataset download script, run one command to evaluate a chosen tool or combination on a chosen split, one command for the cost measurements, one command to rebuild the tables and charts from the saved results. The committed results are the ones we publish, and rerunning on the same machine type should reproduce them within normal timing variation. Model files are downloaded at install time; datasets at run time.

## Results we will produce

Shapes only. Cells are filled by the evaluation run, never by hand.

**Table 1: Detection by tool, overall.** Rows are tools and combinations; columns are precision, recall and F2, each strict and relaxed, for TAB and for MAPA.

| Tool or combination | P (strict) | R (strict) | F2 (strict) | P (relaxed) | R (relaxed) | F2 (relaxed) |
|---|---|---|---|---|---|---|
| (one row per tool and combination) | | | | | | |

**Table 2: F2 by PII type.** Rows are tools; columns are PERSON, ORGANIZATION, LOCATION, DATE_TIME, QUANTITY, IDENTIFIER (TAB only); one table for strict, one for relaxed.

**Table 3: TAB official metrics.** Rows are tools and combinations; columns ER_di, ER_qi, WP, plus any-annotator F2.

**Table 4: Co-reference off versus on.** Rows are detectors; columns are F2 and entity-level recall off, on, and the change.

**Table 5: By language (MAPA).** Rows are tools; columns are languages; cells are F2, with TAB as the English reference column.

**Table 6: Cost and speed.** Rows are tools and combinations; columns are peak memory, model load time, throughput at each input size, and latency per 1,000 characters.

**Chart A: Recall against precision.** One point per tool or combination, with F2 contours drawn on the plot, so the recall-first weighting is visible.

**Chart B: F2 by PII type.** Grouped bars, one group per type, one bar per tool.

**Chart C: Strict versus relaxed.** Paired bars per tool; the gap is the boundary errors.

**Chart D: Accuracy against cost.** F2 on one axis, latency per 1,000 characters on the other, point size for memory.

**Chart E: By language.** Lines or a heat map of F2 across languages for each tool.

**Chart F: Co-reference effect.** Before and after F2 per detector.

## How the results feed the article

Section names are from "A CTO's guide to PII redaction".

| Output | Article section | What it supports |
|---|---|---|
| Table 1, Chart A | Why redaction cannot be your main safeguard | Detection is hard in real free text: even the best tool or combination leaves a measured share of PII unmasked, and the table says how much |
| Table 2, Chart B | Why redaction cannot be your main safeguard | Which kinds of PII are hard (names versus structured values), and that Presidio's patterns are regular expressions, not a full answer |
| Table 3 | Why redaction cannot be your main safeguard | The recognised benchmark's own measures, so the claim does not rest on our scoring alone |
| Chart C | Why redaction cannot be your main safeguard | Finding a name but cutting it short still leaks part of it |
| Table 4, Chart F | Why redaction cannot be your main safeguard; What to do (item 5, redaction as one layer) | Whether linking name forms closes leaks, and how much |
| Table 5, Chart E | Why redaction cannot be your main safeguard | How detection differs across languages |
| Table 6, Chart D | What is still open (cost) | The article says no published production figures exist for what detection costs to run or how much delay it adds; these are measured figures on a stated machine, and they sit next to accuracy |
| Session design | What to do (item 6, describe each control) | A concrete, checkable description of what a redaction step does: "prompts are passed through a redaction step" rather than "we remove all PII"; also that the mapping is held in memory and disposed |
| Deployment diagram | What to do (items 1 and 5) | Where the redaction layer sits relative to the vendor call |

The cost results are stated for one machine and one set of tool versions; the article says so each time it quotes them.

## Deployment diagram (placeholder)

An AWS architecture diagram, a diagram only, with no infrastructure-as-code. The user will produce it. At a high level it will show redaction running inside the company's own AWS environment, before any call to an AI vendor, so that only redacted text crosses the boundary, and the session store that holds the placeholder mapping inside that environment for the life of a session. The details are not designed here.

