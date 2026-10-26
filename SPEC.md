# pii spec

Every requirement is a row with an id. Agents build against approved rows; commits cite the ids they serve. `pullboard spec check` lints this file.

Statuses: approved = decided · draft = proposed · pending = open question · fact = true today · wont = won't build, decided and kept · retired = no longer applies, kept. Ids are permanent: never delete or renumber a row. Tiers: must · aim.

Write each requirement as one row under its section, in this format. The fence keeps the example out of the spec, so no placeholder id is ever committed; your own rows go below, outside it.

```text
- G1 [draft, must] The first thing the client needs, in one line. | gate: the test that proves it
- K1 [draft, must] A constraint, e.g. runs locally with no account. | gate: review
```

## G · Goals: what the client asked for
- G1 [approved, must] Public repo of local PII detection, evaluation and session tools, with reproducible measured results. | gate: review: README states purpose and links the article
- G2 [approved, must] Every number the article quotes comes from code in the repo; nothing is computed by hand. | gate: review: every Table 1-6 and Chart A-F cell traces to a saved result file
- G3 [approved, must] Worldwide English NER (stanfordnlp/en-worldwide-newswire) test split is downloaded at run time and loaded with its source region; Person, Organization, Location, Date and Money map to PERSON, ORGANIZATION, LOCATION, DATE_TIME and QUANTITY; Facility, NORP, Misc and Product are left out. | gate: review
- G4 [approved, must] Every measured setup is scored on the Worldwide English NER test split with co-reference off and on, strict and relaxed, from the test split only. | gate: review
- G5 [approved, must] A table reports person-name precision, recall and F2 per tool on Worldwide English NER, overall and per source region. | gate: review

## K · Constraints
- K1 [approved, must] No text is ever sent to a hosted model or any network service during detection. | gate: test: detection and evaluation pass with network blocked after install and download
- K2 [approved, must] Every tool runs on the reference machine: Apple M5, 32 GB memory, 10 cores. | gate: cost run: every tool loads and runs within 32 GB on the reference machine
- K3 [approved, must] The repo holds no copy of TAB or MAPA; a script downloads them at run time. | gate: test: repo contains no dataset files; download script fetches TAB and MAPA
- K4 [approved, must] Committed results hold aggregate scores only, never dataset text or per-document predictions. | gate: review: committed results hold only aggregate scores
- K5 [approved, must] Model files are downloaded at install time, not committed. | gate: review: install step fetches every model file
- K6 [approved, must] The repo is published under the MIT licence. | gate: review: LICENSE file present
- K7 [approved, must] The repo contains no infrastructure code; the person draws the AWS deployment diagram separately. | gate: review: repo has no infrastructure code

## D · Detection tools
- D1 [approved, must] All detectors share one interface: document in, list of typed character spans out. | gate: test: every detector returns spans with 0 <= start < end <= len(text)
- D2 [approved, must] Detectors and combinations are chosen by configuration from a fixed list of names. | gate: test: unknown detector name in config fails at startup naming the known ones
- D3 [approved, must] Presidio pattern and checksum recognisers cover emails, cards, IBANs, phones, national and tax IDs, IPs, URLs. | gate: test: card with bad check digit not flagged; valid one flagged
- D4 [wont, must] Presidio's structured recognisers always run, in every tool and combination. | gate: review: every run config includes the Presidio structured recognisers
- D5 [approved, must] GLiNER multi_pii-v1, RoBERTa-large-ner-english, spaCy en_core_web_lg and en_core_web_sm each run as a Presidio recogniser. | gate: test: each model detector finds PERSON in a fixed synthetic sentence
- D6 [approved, must] RoBERTa's MISC label is ignored. | gate: test: RoBERTa MISC spans are dropped
- D7 [approved, must] Long text is cut into overlapping segments; spans are mapped back to full-text positions. | gate: test: 100,000-char text yields spans at correct positions across segment joins
- D8 [wont, must] A local quantised language model, via llama.cpp or MLX, lists PII as structured output. | gate: test: LLM detector returns typed spans on a synthetic passage via local runtime
- D9 [wont, must] LLM items are matched back to character positions, tolerating quote and whitespace differences. | gate: test: items with altered quotes or whitespace still map to correct positions; unmatched items reported
- D10 [wont, must] When the language model returns invalid structured output, the call counts as a failure and the run continues. | gate: test: invalid model output is counted and yields no spans, not a crash
- D11 [approved, must] A union masks a span if any member found it; overlapping spans merge into the widest. | gate: test: overlapping spans from two detectors merge to the widest span
- D12 [approved, must] A merged span takes the type and label of its longest contributor, ties broken by higher score. | gate: test: merged span takes type of the longest contributor, ties by higher score
- D13 [approved, must] Measured setups: Presidio alone, each model tool alone, and each model tool with Presidio. | gate: review: config list has Presidio alone, each model alone, each model with Presidio, and no model-on-model unions
- D14 [approved, must] An English-only detector is shown as not applicable for other languages, not scored as missing everything. | gate: test: a detector reports only languages it supports; others show n/a in Table 5
- D15 [wont, must] Agents shortlist and score language models on dev splits; the person picks the model and quantisation. | gate: review: comparison table committed and person's pick recorded
- D16 [approved, must] A pluggable list of names to always recognise: every exact, whole-word occurrence of a listed name is detected as a person, even when no model finds it. Configuration chooses capitalised-only matching (the default; 'Rose' matches, 'rose' does not) or case-insensitive matching. No fuzzy or typo matching. | gate: test: a listed name no model finds is detected; capitalised-only skips 'rose' for listed 'Rose' and case-insensitive matches it; a one-letter typo is never matched
- D17 [approved, must] Local language models of 8B to 12B parameters are not a redaction option: on the reference Mac they take about 1 s per PII item found. | gate: review: no language model in the config list, detector registry or results

## R · Co-reference (identity resolution)
- R1 [approved, must] A Maverick-based co-reference step groups mentions of the same person, names only. | gate: test: "David Parker", "Dave", "Mr. Parker" in one passage share one group
- R2 [approved, must] Co-reference never creates a person: a group is used only if it holds a detected name. | gate: test: a group with no detected name is discarded
- R3 [approved, must] Co-reference masks name forms the detector missed when another form of that person was caught. | gate: test: an undetected form of a detected person gets masked
- R4 [approved, must] Co-reference runs in a separate worker process and returns only character positions, never text. | gate: test: worker reply contains only integers and labels
- R5 [approved, must] Every detector and the best combination are measured with co-reference off and on. | gate: review: every evaluation config runs with co-reference off and on
- R6 [approved, must] Co-reference is measured on English text only (TAB and MAPA English); Maverick is English-only. | gate: review: Table 4 has English rows only
- R7 [approved, must] Co-reference also links name forms through a pluggable nickname list in which every nickname points to exactly one standard given name. The default list is built from carltonnorthern/nicknames (Apache-2.0, its notice kept beside the file) plus added nicknames; configuration can replace it with another file of the same shape or turn it off. | gate: test: 'Dave' joins the 'David Parker' group; 'Matt' and 'Matty' both map to 'Matthew', and neither ever restores to the other; a configured replacement list is used instead of the default

## S · Session: swapping PII out and back in
- S1 [approved, must] One session per case; placeholders like <PERSON_1> count each label within the session. | gate: test: two texts in one session share numbering; a new session restarts at 1
- S2 [approved, must] The same original always gets the same placeholder across every text masked in the session. | gate: test: same original in two texts of one session gets the same placeholder
- S3 [approved, must] With co-reference on, name forms of one person share a placeholder; restore gives the fullest form. | gate: test: with co-reference on, "Dave" and "David Parker" share a placeholder that restores to "David Parker"
- S4 [approved, must] The placeholder mapping lives in process memory only; never written to disk or logs. | gate: test: a run with file and log capture contains no original value
- S5 [approved, must] Logs record placeholders and counts, never original values. | gate: test: logs show placeholders and counts only
- S6 [approved, must] When a session ends, normally or on error, its mapping is deleted. | gate: test: after close, and after an exception in the with-block, the mapping is empty
- S7 [approved, must] Restore tolerates dropped brackets and changed case, matching whole tokens only. | gate: test: <PERSON_1>, PERSON_1 and person_1 restore; PERSON_1 does not match inside PERSON_10
- S8 [approved, must] Restore swaps only placeholders minted in this session; look-alikes pass through untouched. | gate: test: TEAM_7 passes through restore unchanged
- S9 [approved, must] Placeholder-shaped tokens with a minted label that don't resolve are reported as leftovers. | gate: test: <PERSON_9> unminted in a session with PERSON reported as leftover
- S10 [approved, must] A runnable round-trip example uses a stand-in model function that echoes and edits the text. | gate: test: running the example prints masked text, edited reply and restored text
- S11 [approved, must] The repo ships only the in-memory placeholder store. | gate: review: no other store implementation in the repo

## E · Datasets and scoring
- E1 [approved, must] TAB is loaded with its splits, annotators, DIRECT/QUASI/NO_MASK labels and co-reference ids. | gate: test: TAB loader yields 1,014/127/127 train/dev/test documents with annotators kept separate
- E2 [approved, must] MAPA is loaded from Hugging Face with spans rebuilt from BIO tags, per language. | gate: test: MAPA BIO tags rebuild into the expected spans on a fixed sample
- E3 [approved, must] Both datasets and all tool outputs map to PERSON, ORGANIZATION, LOCATION, DATE_TIME, QUANTITY, IDENTIFIER. | gate: test: each mapping row in the brief's table maps as stated
- E4 [approved, must] TAB DEM and MISC are extra rows, left out of the shared-type score; IDENTIFIER is TAB only. | gate: test: TAB DEM and MISC appear as extra rows and not in the shared overall
- E5 [approved, must] F2 (beta 2) is the headline, with precision and recall, per type and overall. | gate: test: hand-worked example gives known P, R and F2
- E6 [approved, must] Matching is greedy and one-to-one; unmatched predictions are false positives, unmatched answers false negatives. | gate: test: one prediction overlapping two answers is credited once
- E7 [approved, must] Every table shows strict (exact start and end) and relaxed (any overlap) matching. | gate: test: a span off by one character counts under relaxed but not strict
- E8 [approved, must] Overall is reported both micro (pooled counts) and macro (average of per-type scores). | gate: test: overall micro and macro differ on an unbalanced fixture as hand-worked
- E9 [approved, must] TAB any-annotator key: a span is PII if any annotator marked it DIRECT or QUASI. | gate: test: answer key keeps distinct annotator boundaries; DEM-only NO_MASK spans excluded
- E10 [approved, must] TAB official ER_di, ER_qi and WP are computed with TAB's own definitions. | gate: test: scores on TAB's published sample match TAB's own scoring script
- E11 [approved, must] Every choice (thresholds, model pick, labels) is fixed on TAB dev and MAPA validation. | gate: review: config and selection code read only dev or validation splits
- E12 [approved, must] Every reported number comes from the test split. | gate: review: every published number comes from TAB test and MAPA test
- E13 [approved, must] MAPA results are broken out by language, English, French and German as headline languages. | gate: test: Table 5 has a column per MAPA language with en, fr, de first

## C · Cost and speed
- C1 [approved, must] Cost is measured per tool and combination, each in a fresh process. | gate: test: cost run starts a fresh process per tool and reports its peak memory
- C2 [approved, must] Cost reports peak memory, load time, throughput at about 1,000/10,000/100,000 chars, median latency per 1,000. | gate: test: cost output has peak memory, load seconds, throughput at 1k/10k/100k, median latency
- C3 [approved, must] Latency uses the median of 5 timed runs after one warm-up run. | gate: review: latency is the median of 5 timed runs after 1 warm-up run
- C4 [approved, must] Tools need not fit AWS Lambda; the cost table shows whether each fits Lambda's 10 GB memory. | gate: review: Table 6 has a fits-Lambda column

## O · Outputs and reproducibility
- O1 [approved, must] Each run writes an environment file: machine, OS, runtime and library versions. | gate: test: a run writes an environment file with machine, OS, runtime and library versions
- O2 [approved, must] A reader reruns with: install, download data, evaluate a tool on a split, measure cost, rebuild tables. | gate: review: README commands run in order on a clean checkout
- O3 [approved, must] Tables 1-6 and Charts A-F are rebuilt by one command from saved results only. | gate: test: rebuild from saved results reproduces committed tables byte for byte
- O4 [approved, must] Chart A plots recall against precision per tool, with F2 contours drawn. | gate: test: Chart A file shows F2 contours
- O5 [approved, must] Tables 1-6 and Charts B-F follow the shapes named in the plan. | gate: review: each table and chart matches its shape in the brief
- O6 [approved, must] Only ideas are taken from poc-anonymizer; no code is copied into this repo. | gate: review: no files or functions copied from poc-anonymizer
