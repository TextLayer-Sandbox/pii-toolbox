# How redaction works

## The idea

Each case (a document, a ticket, a conversation) gets its own session. Before any text leaves the machine, the session swaps every piece of personal information for a typed, numbered placeholder such as `<PERSON_1>` or `<EMAIL_ADDRESS_1>`. The same original always gets the same placeholder, so a model reading the text can still tell that two mentions are the same person. Placeholders are swapped back at one boundary, where the model's answer comes back. The table of placeholders and originals lives in memory only. It is never written to disk or to logs, and it is thrown away when the session closes.

## Terms

- **PII**: personally identifiable information. Any piece of text that can be tied to a real person: a name, an email address, a phone number, a date of birth, an address.
- **Span**: a stretch of the text, given as a start position and an end position, with a type. Positions count characters from the beginning of the text. "David Parker" at characters 0 to 12 is a span of type PERSON.
- **Type and label**: the type is one of six broad kinds (person, organisation, location, date and time, quantity, identifier) that every tool is scored on. The label is the tool's own finer name for what it found, such as EMAIL_ADDRESS or IBAN_CODE. Placeholders show the label, so the masked text still says what kind of value was hidden.
- **Detector**: a tool that reads text and returns spans. One detector may be a set of patterns, another a trained model. They all look the same from outside.
- **Union**: running several detectors and keeping every span any of them found. When two detectors find overlapping spans, the two are merged into one span covering the widest stretch.
- **Segment**: a piece of a long text. Some models can only read a limited amount at once, so the detector cuts the text into overlapping pieces, runs each, and puts the spans back together at their positions in the full text.
- **Identity resolution**: working out which mentions in a text refer to the same person. "David Parker", "Mr. Parker" and "Dave" in one passage are three mentions of one person. The tool that does this is the resolver. The usual name for the technique is co-reference resolution.
- **Entity group**: the identity resolver's output. One group per person, holding every span that refers to that person. One span in the group is the canonical one: the fullest form of the name, such as "David Parker" rather than "Dave".
- **Placeholder**: what replaces a piece of PII in the text. It has a label and a number, written as `<EMAIL_ADDRESS_1>`. The number counts how many of that label the session has seen.
- **Mint**: create a new placeholder for an original value the session has not seen before. "Find or mint" means: give me the placeholder for this value, creating one if there is none yet.
- **Alias**: make a second original value point at an existing placeholder. "Dave" is aliased to the placeholder minted for "David Parker", so both mask to the same thing and restoring gives back the fullest form.
- **Store**: the table from placeholder to original value, kept in memory for the life of one session and deleted when it closes.
- **Session**: the masking context for one case (a document, a ticket, a conversation). Every text masked in the session shares one store, so the same original always gets the same placeholder across all of it.
- **Mask**: replace the PII in a text with placeholders. **Restore**: put the originals back in place of the placeholders.
- **Token**: a placeholder as it appears in text that came back from a model, which may have mangled it slightly. `<PERSON_1>`, `PERSON_1` and `person_1` are all tokens that read as the same placeholder. A "whole token" match means `PERSON_1` is not found inside `PERSON_10`.
- **Look-alike**: a token that has the shape of a placeholder but was never minted in this session, such as an internal handle `TEAM_7`. It is left untouched.
- **Leftover**: a token that reads as a placeholder with a label this session has minted, but that does not match any placeholder in the store. It usually means the model changed a number or invented a placeholder. Leftovers are reported so the problem is visible.
- **Boundary**: the single place in the calling code where masked text goes out to the model and the model's answer is restored on the way back. Everything on the far side of it only ever sees placeholders.
- **ISO 639-1 code**: the standard two-letter code for a language, such as `en` for English, `fr` for French, `de` for German.

## The pieces

- **Detector**: text in, a list of typed spans out. A span is a start, an end and a type.
- **DetectorUnion**: several detectors run together. A hit from any one counts. Overlapping hits are merged into the widest one.
- **IdentityResolver**: looks at the spans and groups the ones that are the same person, such as "Dave" and "David Parker".
- **PlaceholderFormat**: the one definition of what a placeholder looks like in text. It writes `<PERSON_1>` and reads back tokens that were lightly mangled.
- **PlaceholderStore**: remembers, per session, which placeholder stands for which original.
- **RedactionSession**: ties the pieces together. It can `mask` text, `restore` text and `close`.

## Step by step through `mask(text)`

1. Wrap the text in a `Document`. Its id is the session id plus a counter. Its language is the session's fixed language.
2. Call `detector.detect(document)` to get spans. If the detector is a `DetectorUnion`, the merging has already happened inside it. If the text was too long for a model, the detector that needs it has already cut it into segments and put the pieces back together. The session sees none of that.
3. If a resolver is set, call `resolver.resolve(document, spans)` to get `EntityGroup`s. For each group, take the canonical span's text and call `get_or_mint` for it, which returns the placeholder and whether this call minted it. Then call `alias` for the text of every other span in the group, pointing at the same placeholder. This is why "Dave" and "David Parker" both become `<PERSON_1>`, and why the stored original is the fullest form of the name.
4. For every span not in a group, call `get_or_mint(session_id, span.detector_label, original_text)`. The label is the detector's detailed label (`EMAIL_ADDRESS`, not `IDENTIFIER`), so the masked text still says what kind of value was hidden.
5. Replace the spans in the text from the end backwards, so earlier positions stay valid. Each placeholder is written with `format.render`, for example `<LABEL_N>`.
6. Return a `MaskResult` with the masked text and the placeholders that `get_or_mint` reported as minted by this call.

```python
def mask(self, text: str) -> MaskResult:
    self.counter += 1
    document = Document(f"{self.id}-{self.counter}", text, self.language)
    spans = self.detector.detect(document)

    placeholder_for: dict[Span, Placeholder] = {}
    minted: list[Placeholder] = []

    def get_or_mint(label: str, original: str) -> Placeholder:
        placeholder, was_minted = self.store.get_or_mint(self.id, label, original)
        if was_minted:
            minted.append(placeholder)
        return placeholder

    if self.resolver is not None:
        for group in self.resolver.resolve(document, spans):
            fullest = text[group.canonical.start:group.canonical.end]
            placeholder = get_or_mint(group.canonical.detector_label, fullest)
            for span in group.spans:
                if span != group.canonical:
                    self.store.alias(self.id, text[span.start:span.end], placeholder)  # same person, same placeholder
                placeholder_for[span] = placeholder

    for span in spans:
        if span not in placeholder_for:
            original = text[span.start:span.end]
            placeholder_for[span] = get_or_mint(span.detector_label, original)

    masked = text
    for span in sorted(placeholder_for, key=lambda s: s.start, reverse=True):  # end first, so positions hold
        masked = masked[:span.start] + self.format.render(placeholder_for[span]) + masked[span.end:]

    return MaskResult(masked, tuple(minted))
```

A plain detector can be very small. This one finds email addresses with a pattern:

```python
class EmailDetector(Detector):
    def detect(self, document: Document) -> list[Span]:
        return [
            Span(m.start(), m.end(), PiiType.IDENTIFIER, "EMAIL_ADDRESS", (self.name,), None)
            for m in re.finditer(r"[\w.]+@[\w.]+\.\w+", document.text)
        ]
```

## Sending and masking what comes back

Only the masked text goes to the model. The model may ask for more data, for example through a tool that reads a database row. Whatever a backend returns is passed through `mask` again, on the same session, before the model sees it. A value the session already knows gets its existing placeholder. A new value gets a new one.

```python
masked = session.mask("Email David Parker at david@acme.com")
reply = model(masked.masked_text)

row = database.lookup("david@acme.com")      # real data: contains the real name
safe_row = session.mask(row)                 # known email keeps its placeholder; a new phone number gets one
model(safe_row.masked_text)
```

## Step by step through `restore(text)`

1. Call `format.parse(text)` to find tokens that look like placeholders. It is tolerant of light mangling: `<PERSON_1>`, `PERSON_1` and `person_1` all count.
2. Parsing matches whole tokens only, so `PERSON_1` is never found inside `PERSON_10`.
3. Look each one up with `store.original`.
4. Substitute the ones that resolve, working from the last token to the first so earlier positions stay valid.
5. A token that does not resolve is a leftover only when its label is one this session has minted (`store.labels`). It usually means the model mangled or invented a token.
6. Only placeholders minted in this session are swapped. A look-alike such as an internal handle `TEAM_7` passes through untouched and is not reported.

```python
def restore(self, text: str) -> RestoreResult:
    minted_labels = self.store.labels(self.id)
    leftovers = []
    restored = text

    for token in sorted(self.format.parse(text), key=lambda t: t.start, reverse=True):
        original = self.store.original(self.id, token.placeholder)
        if original is not None:
            restored = restored[:token.start] + original + restored[token.end:]
        elif token.placeholder.label in minted_labels:
            leftovers.append(text[token.start:token.end])  # reported, and left in the text as it was

    return RestoreResult(restored, tuple(reversed(leftovers)))
```

## Closing

`close` calls `store.discard(session_id)`, which deletes every mapping for the session. Using the session in a `with` block closes it even if the block raises an error.

Here is the whole round trip. The stand-in model echoes the text and edits it.

```python
def fake_model(text: str) -> str:
    return "Summary: " + text.replace("wrote", "emailed")

with session:                                                 # closed on exit, even on error
    masked = session.mask("David Parker wrote to dave@acme.com. Dave replied.")
    # masked.masked_text == "<PERSON_1> wrote to <EMAIL_ADDRESS_1>. <PERSON_1> replied."

    answer = fake_model(masked.masked_text)                   # the only place text leaves
    result = session.restore(answer)

print(result.restored_text)  # Summary: David Parker emailed dave@acme.com. David Parker replied.
print(result.leftovers)      # ()
```

The `with` block is the boundary. Everything before `fake_model` is masked. Everything after `restore` is real data.

## Decisions recorded in the shapes

- **Language.** The session carries one fixed language, given when it is opened, and every text it masks is taken to be in that language. Detecting the language of each text is a separate, later concern.
- **Placeholder text form.** How a placeholder looks in text lives in one shared `PlaceholderFormat`, with `render` and a tolerant `parse`. Both `mask` and `restore` use it, so the two can never disagree.
- **The resolver adding spans.** A group may hold a span the detector missed, and that span's `source` names the resolver. A group is still never returned unless it holds at least one detected name.
- **Building a session.** A session is built from a detector, an optional resolver, a store, a placeholder format and a language. `DetectorRegistry` builds the detector from configuration.
- **The boundary.** "One boundary" is a rule for calling code, not a type. It is the one place that masks before the call and restores after, as the round trip above shows.
- **Leftovers.** A leftover is only a token whose label this session has minted, so look-alikes with unknown labels are never reported.
