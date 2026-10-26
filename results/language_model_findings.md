# Language-model cost findings

Decision D17 rules out local language models of 8B to 12B parameters as a redaction option. This note keeps what was measured. It holds aggregate numbers only and no dataset text.

## Reference machine and setup

- Apple M5 with 32 GB of memory.
- Qwen3 8B at 4-bit: a Q4_K_M GGUF file run through llama.cpp, and an MLX conversion run through MLX.
- The same 4 TAB dev passages in every run, 11,995 characters in total.
- The raw timings are in `results/language_model_findings/runtime_timing_before_fix.json` and `results/language_model_findings/runtime_timing_after_fix.json`, unchanged from the commits that produced them.

## Round 1: adapters as first built

| Runtime | Load | Time per 1,000 characters |
| --- | --- | --- |
| llama.cpp | 6.9 s | 17.75 s |
| MLX | 3.4 s | 26.87 s |

llama.cpp ran on the CPU only, with no GPU offload. MLX was slow because Qwen3 wrote about 7,000 characters of reasoning before its JSON. With GPU offload switched on but no chat template and no repeat penalty, llama.cpp looped to the 2,048-token cap on 3 of the 4 passages.

## Round 2: after the runtime fix

The fix was GPU offload, the chat template, reasoning switched off and a repeat penalty of 1.1.

| Runtime | Load | Time per 1,000 characters | Valid output |
| --- | --- | --- | --- |
| llama.cpp | 0.55 s | 9.67 s per 1,000 characters | 4 of 4 |
| MLX | 3.05 s | 6.68 s per 1,000 characters | 2 of 4 |

MLX had no grammar constraint, which is why only half of its outputs were valid.

## Where the llama.cpp time goes

Per 1,000 characters, 9.72 s measured:

| Part | Detail | Time | Share |
| --- | --- | --- | --- |
| Generation | about 182 output tokens at about 23 tokens per second | 7.92 s | 81% |
| Grammar sampler overhead | about 7.4 ms per token | 1.35 s | 14% |
| Prefill | about 650 tokens at about 570 tokens per second | 0.46 s | 5% |
| Load, grammar compile, prompt-prefix reuse | | about 0 s | 0% |

## Cost per item found

A 3,000-character passage gives 20 to 38 PII items and 384 to 690 output tokens. That is about 19 output tokens per item: about 55% JSON scaffolding, 35% echoed entity text and 10% the type. The cost is about 0.8 s per item found.

## Hardware limits

All 37 layers were offloaded to Metal. Generation without a grammar ran at 24.8 tokens per second, close to the memory-bandwidth limit for a 4.8 GB model. Flash attention, a batch size of 2,048 and the thread count changed nothing beyond noise.

## Full-run estimates

On the dev data of 787,000 characters:

- Qwen3 8B through llama.cpp: about 2.1 hours.
- All four shortlisted models (Qwen3 8B, Gemma 3 12B, Mistral Nemo 12B, Llama 3.1 8B): about 11 to 12 hours.

The 12B figures were estimated from the 8B timing and were not measured.

## Conclusion

About 1 s of generation per PII item found makes this approach too slow for bulk redaction on this hardware, and larger models are slower still. D17 therefore removes language models from the options.
