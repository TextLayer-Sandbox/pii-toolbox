"""Adapters that run a local quantised language model through llama.cpp or MLX.

Each adapter loads a model from a local path and answers prompts under a JSON schema.
The runtime libraries are imported when an adapter is built so the package imports on
machines that have neither installed.
"""

import importlib
import json
from pathlib import Path
from typing import Any

MAX_NEW_TOKENS = 2048
CONTEXT_TOKENS = 8192


def _require_model_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"model path does not exist: {path}")
    return path


class LlamaCppRuntime:
    """Runs a GGUF model with llama.cpp, constraining output with a grammar built from the schema."""

    def __init__(self, model_path: str | Path) -> None:
        path = _require_model_path(model_path)
        llama_cpp = importlib.import_module("llama_cpp")
        self._grammar_class = llama_cpp.LlamaGrammar
        self._model = llama_cpp.Llama(model_path=str(path), n_ctx=CONTEXT_TOKENS, verbose=False)

    def generate_structured(self, prompt: str, json_schema: dict[str, Any]) -> str:
        grammar = self._grammar_class.from_json_schema(json.dumps(json_schema))
        completion = self._model.create_completion(
            prompt=prompt, grammar=grammar, max_tokens=MAX_NEW_TOKENS, temperature=0.0
        )
        return completion["choices"][0]["text"]


class MlxRuntime:
    """Runs a model with mlx_lm, which has no grammar support, so the schema is stated in the prompt."""

    def __init__(self, model_path: str | Path) -> None:
        path = _require_model_path(model_path)
        self._mlx_lm = importlib.import_module("mlx_lm")
        self._model, self._tokenizer = self._mlx_lm.load(str(path))

    def generate_structured(self, prompt: str, json_schema: dict[str, Any]) -> str:
        constrained_prompt = f"{prompt}\n\nReply with only JSON that matches this schema:\n{json.dumps(json_schema)}"
        return self._mlx_lm.generate(
            self._model, self._tokenizer, prompt=constrained_prompt, max_tokens=MAX_NEW_TOKENS
        )
