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
# Greedy decoding loops on repeated phrases; 1.1 is the usual mild setting that breaks loops without distorting names.
REPETITION_PENALTY = 1.1


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
        self._model = llama_cpp.Llama(model_path=str(path), n_ctx=CONTEXT_TOKENS, n_gpu_layers=-1, verbose=False)

    def generate_structured(self, prompt: str, json_schema: dict[str, Any]) -> str:
        grammar = self._grammar_class.from_json_schema(json.dumps(json_schema))
        # The grammar forces JSON from the first token, so a reasoning preamble cannot be emitted.
        completion = self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            grammar=grammar,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            repeat_penalty=REPETITION_PENALTY,
        )
        return completion["choices"][0]["message"]["content"]


class MlxRuntime:
    """Runs a model with mlx_lm, which has no grammar support, so the schema is stated in the prompt."""

    def __init__(self, model_path: str | Path) -> None:
        path = _require_model_path(model_path)
        self._mlx_lm = importlib.import_module("mlx_lm")
        self._model, self._tokenizer = self._mlx_lm.load(str(path))
        sample_utils = importlib.import_module("mlx_lm.sample_utils")
        self._logits_processors = sample_utils.make_logits_processors(repetition_penalty=REPETITION_PENALTY)

    def generate_structured(self, prompt: str, json_schema: dict[str, Any]) -> str:
        constrained_prompt = f"{prompt}\n\nReply with only JSON that matches this schema:\n{json.dumps(json_schema)}"
        chat_prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": constrained_prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        return self._mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=chat_prompt,
            max_tokens=MAX_NEW_TOKENS,
            logits_processors=self._logits_processors,
        )
