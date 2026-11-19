# Vocabulary

The conceptual model behind the toolbox. Field-level detail is in `core/types.py`.

## Masking

- **Span**: a stretch of text marked as personal information, with a type.
- **Type**: one of six shared categories (person, organisation, location, date and time, quantity, identifier) that every detector and dataset is mapped onto.
- **Detector**: finds spans. Writing `a+b` makes a union: both run, and overlapping spans merge into one.
- **Form**: one way a name is written, such as "Dr John Smith" or "Smith".
- **Mention**: one span that refers to a person.
- **Group** (`PersonMentions`): the mentions in one text that refer to the same person. Its fullest form is the **canonical** one. Only people have groups.
- **Identity resolver**: builds the groups for a text. It can add name mentions the detector missed, but never a person the detector did not find. The co-reference model's raw output is a **cluster**: every reference to one thing, including pronouns such as "he" and descriptions such as "the applicant". A group keeps only the names from a cluster.
- **Placeholder**: what a span is replaced with, such as `<PERSON_1>`. One placeholder stands for one value, or one person, for the whole session.
- **Alias**: a shorter form that shares the placeholder of a fuller form.
- **Session**: one redaction session. Everything masked through it shares placeholders, and the mapping is discarded when it closes.

A session masks one text at a time, such as each message of a conversation. A group is worked out for a single text and then forgotten; a placeholder is remembered until the session closes. So for each new text the session decides whether a group is a person it already knows: it reuses a placeholder when one side's fullest form is among the other side's forms. Restoring always gives the fullest form.

## Measuring

- **Setup**: a detector or union being measured.
- **Answer**: a dataset's annotation of where the personal information is. Overlapping answers of one type form an **answer group** and count once. This is unrelated to a person's group.
- **Strict** and **relaxed**: strict needs the same boundaries; relaxed needs the overlap to cover at least half of the longer span (the code calls it `ANY_OVERLAP`).
- **Micro** and **macro**: micro pools the counts across types; macro averages the per-type scores.
