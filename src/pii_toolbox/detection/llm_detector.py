"""Detector that asks a local language model to list the PII in a passage.

The model returns each item as its exact text and a type name; the items are matched back
to character positions in the passage. Output that does not parse into the expected shape
counts as a failed call and yields no spans, so one bad reply never stops a run.
"""

import json
from typing import Any, Protocol

from pii_toolbox.core import Detector, Document, PiiType, Span
from pii_toolbox.detection.llm_matching import DEFAULT_LABEL_MAP, match_items


class LlmRuntime(Protocol):
    """A loaded local language model that can answer under a JSON schema."""

    def generate_structured(self, prompt: str, json_schema: dict[str, Any]) -> str:
        """Returns the model's reply text, constrained to `json_schema` where supported."""
        ...


PII_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": [pii_type.value for pii_type in PiiType]},
                },
                "required": ["text", "type"],
            },
        }
    },
    "required": ["items"],
}

_PROMPT_TEMPLATE = (
    "List every piece of personally identifying information in the passage below. "
    "For each one give its exact text, copied character for character from the passage, "
    "and its type, which must be one of: {types}. Answer as JSON with an `items` list.\n\n"
    "Passage:\n{text}"
)


def build_prompt(text: str) -> str:
    return _PROMPT_TEMPLATE.format(types=", ".join(pii_type.value for pii_type in PiiType), text=text)


def parse_items(reply: str) -> list[tuple[str, str]]:
    """Returns (text, type) pairs from a reply, raising ValueError when the shape is wrong."""
    try:
        payload = json.loads(reply)
    except json.JSONDecodeError as error:
        raise ValueError("reply is not JSON") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("reply has no items list")
    items: list[tuple[str, str]] = []
    for entry in payload["items"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str) or not isinstance(entry.get("type"), str):
            raise ValueError("an item lacks text or type strings")
        items.append((entry["text"], entry["type"]))
    return items


class LlmDetector(Detector):
    """Detects PII with any language model runtime.

    `failed_calls` counts replies that could not be parsed; `unmatched_items` counts items
    that could not be found in the passage or carried an unknown type.
    """

    def __init__(self, runtime: LlmRuntime, name: str = "llm") -> None:
        self._runtime = runtime
        self._name = name
        self.failed_calls = 0
        self.unmatched_items = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def languages(self) -> frozenset[str] | None:
        return None

    @property
    def detects(self) -> frozenset[PiiType]:
        return frozenset(PiiType)

    def load_models(self) -> None:
        """The runtime is supplied already loaded."""

    def detect(self, document: Document) -> list[Span]:
        reply = self._runtime.generate_structured(build_prompt(document.text), PII_ITEMS_SCHEMA)
        try:
            items = parse_items(reply)
        except ValueError:
            self.failed_calls += 1
            return []
        spans, unmatched = match_items(document.text, items, DEFAULT_LABEL_MAP, source=self._name)
        self.unmatched_items += len(unmatched)
        return spans
