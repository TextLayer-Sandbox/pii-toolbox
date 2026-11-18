# Masking toolbox

The toolbox replaces personal information in a text with placeholders such as `<PERSON_1>`, and later puts the original values back. This guide covers the four parts of `src/pii_toolbox` that do that work (`core`, `detection`, `coref` and `session`) and walks through the example in `examples/round_trip.py`.

The idea: before text goes to another system (for example a language model), a session finds each piece of personal information and swaps it for a numbered placeholder. The same original value always gets the same placeholder, so the reader can still tell that two mentions are the same person. When the answer comes back, the session swaps the placeholders for the original values. The table of originals is kept in memory only and is thrown away when the session closes.

## Plain terms

- **Span**: a stretch of text with a start position, an end position and a type. Positions count characters from the start of the text; the start is included and the end is not.
- **Type**: one of six broad kinds used for scoring: `PERSON`, `ORGANIZATION`, `LOCATION`, `DATE_TIME`, `QUANTITY` and `IDENTIFIER` (emails, phone numbers, card numbers, IBANs, IP addresses, URLs and similar codes).
- **Label**: a detector's own finer name for what it found, such as `EMAIL_ADDRESS`. Placeholders use the type, so masked text still says what kind of value was hidden.
- **Placeholder**: a type plus a number, written `<TYPE_N>`. The number counts how many of that type the session has created.
- **Store**: the in-memory table from placeholder to original value.

## The four packages

### `core`

Shared types and the abstract interfaces everything else is built on, kept in `src/pii_toolbox/core/types.py`:

- `Span`, `Document` and `PiiType` describe detected text.
- `Detector` takes a `Document` and returns a list of spans. It also says which languages it supports and which types it can report.
- `IdentityResolver` takes a document and its spans and returns groups of spans that refer to the same person (`PersonMentions`). Each group names one span as the canonical one: the fullest form of the name.
- `PlaceholderFormat`, `PlaceholderStore` and `RedactionSession` describe the masking side.

`core/label_maps.py` holds the translation tables from each tool's or dataset's own labels to the six common types.

### `detection`

Detectors, all behind the one `Detector` interface. The registry (`detection/registry.py`) lists them by name:

| Name | What it does |
| --- | --- |
| `presidio_structured` | Pattern and checksum recognisers from Presidio for card numbers, emails, IBANs, IP addresses, URLs, phone numbers and US social security numbers. Reports `IDENTIFIER` only. |
| `spacy_lg`, `spacy_sm` | The large and small English spaCy pipelines. Report person, organisation, location, date and time, and quantity entities. English only. |
| `roberta` | An English RoBERTa entity tagger. Reports person, organisation and location. English only. |
| `gliner` | The multilingual GLiNER PII model, asked for person, organisation, location, address, date, money, case number and identification number (address maps to location, money to quantity, and the two number labels to identifier). Reports person, organisation, location, date and time, quantity and identifier. The money label finds amounts of money, not plain counts. |
| `known_names` | A fixed list of names that are always treated as people. Matches whole words only, and needs no model. |

Two helpers sit alongside them:

- **Union.** Writing `a+b` as a detector name (for example `gliner+presidio_structured`) runs both and keeps every span either one found. Overlapping spans are merged into one covering the widest stretch; the merged span takes the type and label of its longest contributing span. Spans that only touch stay separate.
- **Segmenter.** The model detectors can only read a limited amount of text at once, so long text is cut into overlapping pieces, each piece is detected separately, and the spans are mapped back to positions in the full text.

Model files are read from a model folder (see [Evaluation](evaluation.md) for how to download them). Detectors never download anything while detecting.

### `coref`

Co-reference resolution links the different ways a text refers to one person, such as "David Parker", "Mr. Parker" and "Dave". It is an `IdentityResolver`.

- `MaverickResolver` (`coref/resolver.py`) asks a separate worker process, which runs the Maverick co-reference model in its own `uv` environment (`coref/worker`). The two talk over a local ZeroMQ socket. Only character positions travel back, never text. If the worker does not answer in time it is restarted and the request is sent again, up to a limit. Start it with `start()` or use it as a context manager. It handles English only.
- `groups.py` turns the model's clusters of mentions into groups. A group is only made when it already contains a name the detector found, so the resolver never invents a person. Other mentions in the cluster are added only if they look like a name (a title followed by capitalised words, or all capitalised words) and are not pronouns.
- `nickname_step.py` then adds detected names the model left unlinked, using a nickname list (`coref/nicknames.csv`, for example `dave` goes to `david`). A name joins a group only when exactly one group holds a name with the same standard given name and a compatible surname.

### `session`

`Session` (`session/session.py`) ties the pieces together. It is built from a detector, a placeholder store, a placeholder format, a language code and, optionally, an identity resolver.

- `AngleBracketFormat` writes `<LABEL_N>` and, when restoring, also reads tokens that a model has lightly changed: brackets dropped or letter case changed. A token must stand alone, so `PERSON_1` is not picked out of `PERSON_10`.
- `InMemoryPlaceholderStore` keeps, for each session id, which placeholder stands for which original. It never writes originals to disk or logs. When several forms of a name share one placeholder, restoring gives back the longest form.

**Masking a text** (`Session.mask`):

1. Detect spans with the detector.
2. If a resolver is set, ask it for groups. Each group gets one placeholder, created for the canonical (fullest) form; every other form in the group is stored as an alias of it.
3. Every remaining span gets a placeholder for its own type, reusing an existing one if the same original was seen before.
4. Replace the spans in the text with the rendered placeholders.

The result is a `MaskResult` holding the masked text and the placeholders created by this call.

**Restoring a text** (`Session.restore`): find every token written exactly as `<LABEL_N>` (upper-case label, angle brackets, no leading zeros) and replace the ones this session knows with their originals. A token that looks like a placeholder, with a type this session has created but a number it does not know, is left in place and listed in `leftovers`, so a changed number is visible. Look-alikes with a type the session never created are ignored.

**Closing** (`Session.close`, or leaving a `with` block) deletes everything the store holds for the session.

## Walk-through of `examples/round_trip.py`

Run it from the repository root:

```
uv run python examples/round_trip.py
```

It uses no models, so it runs straight away. The script defines three small pieces and then uses them.

1. `NameAndEmailDetector` is a `Detector`. It reports the fixed names "David Parker" and "Dave" as `PERSON` and anything shaped like an email address as an `IDENTIFIER` with the label `EMAIL_ADDRESS`.
2. `SinglePersonResolver` is an `IdentityResolver` that treats every detected person in a text as one individual, with the longest form as the canonical name.
3. `stand_in_model` plays the part of a language model: it adds a `Summary: ` prefix and changes "wrote to" to "emailed".
4. `main()` builds a `Session` from the detector, an `InMemoryPlaceholderStore`, an `AngleBracketFormat`, the language `"en"` and the resolver. Inside a `with` block it masks the text `David Parker wrote to dave@acme.example. Dave replied.`, passes the masked text to the stand-in model, and restores the reply.

Output:

```
masked: <PERSON_1> wrote to <IDENTIFIER_1>. <PERSON_1> replied.
reply: Summary: <PERSON_1> emailed <IDENTIFIER_1>. <PERSON_1> replied.
restored: Summary: David Parker emailed dave@acme.example. David Parker replied.
```

Points to notice:

- "David Parker" and "Dave" both became `<PERSON_1>`, because the resolver grouped them and the session aliased the shorter form to the same placeholder.
- The email address became `<IDENTIFIER_1>`: the placeholder uses the type.
- On restore, both `<PERSON_1>` tokens came back as "David Parker", the fullest form, so the second mention ("Dave") is not restored to its original wording.
- The `with` block closed the session on exit, which discarded the stored originals.

To use real detectors instead, build them by name with `build_from_config` from `pii_toolbox.detection.registry` (for example `build_from_config(["gliner+presidio_structured"])`), after downloading the models. To link name forms in English text, pass a started `MaverickResolver` as the `resolver`.
