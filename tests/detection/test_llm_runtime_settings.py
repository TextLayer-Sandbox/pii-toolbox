import sys
import types

import pytest

from pii_toolbox.detection.llm_runtimes import MAX_NEW_TOKENS, LlamaCppRuntime, MlxRuntime

SCHEMA = {"type": "object", "properties": {"items": {"type": "array"}}}


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"")
    return path


@pytest.fixture
def llama_cpp_calls(monkeypatch):
    calls = {}

    class FakeGrammar:
        @classmethod
        def from_json_schema(cls, schema_text):
            calls["grammar_schema"] = schema_text
            return cls()

    class FakeLlama:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def create_chat_completion(self, **kwargs):
            calls["chat"] = kwargs
            return {"choices": [{"message": {"content": '{"items": []}'}}]}

        def create_completion(self, **kwargs):
            raise AssertionError("raw completion bypasses the chat template")

    module = types.ModuleType("llama_cpp")
    module.Llama = FakeLlama
    module.LlamaGrammar = FakeGrammar
    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    return calls


@pytest.fixture
def mlx_calls(monkeypatch):
    calls = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls["template"] = (messages, kwargs)
            return "<templated prompt>"

    def load(path):
        return "model-object", FakeTokenizer()

    def generate(model, tokenizer, prompt, max_tokens, **kwargs):
        calls["generate"] = {"prompt": prompt, "max_tokens": max_tokens, **kwargs}
        return '{"items": []}'

    def make_logits_processors(**kwargs):
        calls["processor_args"] = kwargs
        return ["penalty-processor"]

    module = types.ModuleType("mlx_lm")
    module.load = load
    module.generate = generate
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_logits_processors = make_logits_processors
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    return calls


def test_llama_cpp_offloads_all_layers_and_uses_chat_template_with_repeat_penalty(model_file, llama_cpp_calls):
    runtime = LlamaCppRuntime(model_file)
    assert llama_cpp_calls["init"]["n_gpu_layers"] == -1
    assert runtime.generate_structured("list pii", SCHEMA) == '{"items": []}'
    chat = llama_cpp_calls["chat"]
    assert chat["messages"] == [{"role": "user", "content": "list pii"}]
    assert chat["grammar"] is not None
    assert chat["repeat_penalty"] > 1.0
    assert chat["max_tokens"] == MAX_NEW_TOKENS
    assert chat["temperature"] == 0.0


def test_mlx_uses_chat_template_without_thinking_and_repetition_penalty(model_file, mlx_calls):
    runtime = MlxRuntime(model_file)
    assert runtime.generate_structured("list pii", SCHEMA) == '{"items": []}'
    messages, template_kwargs = mlx_calls["template"]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("list pii")
    assert template_kwargs["add_generation_prompt"] is True
    assert template_kwargs["enable_thinking"] is False
    assert template_kwargs["tokenize"] is False
    assert mlx_calls["processor_args"]["repetition_penalty"] > 1.0
    generated = mlx_calls["generate"]
    assert generated["prompt"] == "<templated prompt>"
    assert generated["logits_processors"] == ["penalty-processor"]
    assert generated["max_tokens"] == MAX_NEW_TOKENS
