# Runtime smoke test

2026-10-18 · M3 Pro · 36 GB unified memory · macOS 26.7.1 · Python 3.11.15

Target: Qwen/Qwen3.5-9B, Transformers 5.20.1, PyTorch 2.14.1, Metal 4-bit
linear layers (group size 64). Kernels 0.17.2; Hugging Face Hub 2.2.0.

## Result

- MPS available; device arithmetic passed.
- Metal quantization kernel loaded with the standard publisher trust check.
- A reduced, randomly initialized Qwen3.5 model passed four generation steps
  with three linear-attention layers and one full-attention layer on MPS.
  This is an architecture check, not the pretrained 9B model or a quality test.
- The full pretrained model ran both synthetic prompts on MPS. The final output
  passed the strict ten-ID permutation validator. No CPU fallback was enabled.

The resolved model revision is `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
The source checkpoint contains 19.31 GB of safetensors. Quantization happens at
load time; it does not reduce this download.

## HTTP issue

With telemetry disabled, kernels passes an empty custom User-Agent to Hub.
Hub appends a separator and space; the HTTP client rejects the trailing space.
Kernels wraps this as a publisher trust error, hiding the underlying exception.

The smoke script trims trailing User-Agent whitespace through a request hook.
TLS and publisher checks remain enabled. A regression test covers the hook.
An independent Hub API call confirmed the publisher's trusted status.

## Measurement scope

The real-model test uses two synthetic prompts: a short instruction and a ten-item
reranking request. It checks execution and output format, not retrieval quality.

Reports include process peak RSS, sampled MPS allocation, and generation wall time
including prefill and device synchronization. These are not decode-only throughput
or system-wide peak memory. The script does not enable CPU fallback.

DeltaNet uses reference PyTorch implementations. The full model ran on MPS,
but these short inputs do not establish acceptable latency for 4000-token evidence
contexts. Output-format success is not an assessment of biomedical relevance.

## Sources

- [Model](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Apple Silicon](https://huggingface.co/docs/transformers/main/apple_silicon)
- [Metal quantization](https://huggingface.co/docs/transformers/main/quantization/metal)

## Full-model runs

| Run | Input tokens (short / ranking) | Short response | Ranking | Result |
|---|---:|---:|---:|---|
| [003](runtime-reports/inference-003.json) | 23 / 179 | 4.29 s | 5.34 s | Integer positions instead of IDs |
| [004](runtime-reports/inference-004.json) | 23 / 236 | 3.05 s | 3.71 s | Four IDs; incomplete list |
| [005](runtime-reports/inference-005.json) | 23 / 257 | 3.06 s | 7.04 s | Valid permutation |

The first prompt returned integer positions. The second preserved string IDs but
omitted six candidates; its wording no longer explicitly required all ten.
The final prompt states that all ten IDs are required, including irrelevant ones.
Both invalid outputs were rejected, not repaired or counted as successes.

The prompts differ, so these rows are not a controlled speed comparison.
All raw prompts and outputs are linked above. No retrieval-quality metric has
been measured yet.

Final run memory observations:

- Process peak RSS: 13.99 GiB.
- Maximum sampled MPS tensor allocation: 7.60 GiB.
- Maximum sampled MPS driver allocation: 26.62 GiB.

These counters overlap in unified memory; do not add them. Driver allocation
includes allocator/kernel resources and is not a measurement of system-wide
physical RAM use. Sampling interval: 0.2 s. Other applications remained open.

The initial load, including the 19.31 GB download, took
46.0 minutes. The final cached
load took 19.85 s, including metadata
checks and loading-time quantization. No pre-quantized checkpoint was saved.

## Decision

Keep Transformers/MPS for the first experiments. It loads and runs the selected
model on this laptop. Revisit runtime after measuring real evidence-sized inputs.
Listwise output must still be validated; production fallback to RRF is not yet
implemented.
