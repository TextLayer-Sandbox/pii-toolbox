import json
import sys
import types

import pytest

from pii_toolbox.detection.llm_runtimes import LlamaCppRuntime, MlxRuntime

SCHEMA = {"type": "object", "properties": {"items": {"type": "array"}}}


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"")
    return path


@pytest.fixture
def llama_cpp_stub(monkeypatch):
    calls = {}

    class StubGrammar:
        @classmethod
        def from_json_schema(cls, schema_text):
            calls["grammar_schema"] = schema_text
            grammar = cls()
            grammar.schema_text = schema_text
            return grammar

    class StubLlama:
        def __init__(self, model_path, **kwargs):
            calls["model_path"] = model_path

        def create_completion(self, prompt, grammar, **kwargs):
            calls["prompt"] = prompt
            calls["grammar"] = grammar
            return {"choices": [{"text": '{"items": []}'}]}

    stub = types.ModuleType("llama_cpp")
    stub.Llama = StubLlama
    stub.LlamaGrammar = StubGrammar
    monkeypatch.setitem(sys.modules, "llama_cpp", stub)
    return calls


@pytest.fixture
def mlx_lm_stub(monkeypatch):
    calls = {}

    def load(path):
        calls["model_path"] = path
        return "model-object", "tokenizer-object"

    def generate(model, tokenizer, prompt, max_tokens):
        calls["generate"] = (model, tokenizer, prompt)
        return '{"items": []}'

    stub = types.ModuleType("mlx_lm")
    stub.load = load
    stub.generate = generate
    monkeypatch.setitem(sys.modules, "mlx_lm", stub)
    return calls


def test_llama_cpp_passes_schema_as_grammar_and_returns_text(model_file, llama_cpp_stub):
    runtime = LlamaCppRuntime(model_file)
    assert runtime.generate_structured("list pii", SCHEMA) == '{"items": []}'
    assert llama_cpp_stub["model_path"] == str(model_file)
    assert json.loads(llama_cpp_stub["grammar_schema"]) == SCHEMA
    assert llama_cpp_stub["grammar"].schema_text == llama_cpp_stub["grammar_schema"]
    assert llama_cpp_stub["prompt"] == "list pii"


def test_mlx_states_schema_in_prompt_and_returns_text(model_file, mlx_lm_stub):
    runtime = MlxRuntime(model_file)
    assert runtime.generate_structured("list pii", SCHEMA) == '{"items": []}'
    assert mlx_lm_stub["model_path"] == str(model_file)
    model, tokenizer, prompt = mlx_lm_stub["generate"]
    assert (model, tokenizer) == ("model-object", "tokenizer-object")
    assert prompt.startswith("list pii")
    assert json.dumps(SCHEMA) in prompt


@pytest.mark.parametrize("runtime_class", [LlamaCppRuntime, MlxRuntime])
def test_missing_model_path_raises_file_not_found_naming_the_path(runtime_class, tmp_path, llama_cpp_stub, mlx_lm_stub):
    missing = tmp_path / "absent" / "model.bin"
    with pytest.raises(FileNotFoundError, match="absent"):
        runtime_class(missing)


def test_importing_the_module_needs_neither_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    import importlib

    import pii_toolbox.detection.llm_runtimes as module

    importlib.reload(module)
