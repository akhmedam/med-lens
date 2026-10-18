# Runtime smoke test

2026-10-18 · M3 Pro · 36 GB unified memory · macOS 26.7.1 · Python 3.11.15

Target: Qwen/Qwen3.5-9B, Transformers 5.20.1, PyTorch 2.14.1, Metal 4-bit
linear layers (group size 64). Kernels 0.17.2; Hugging Face Hub 2.2.0.

## Checks so far

- MPS available; device arithmetic passed.
- Metal quantization kernel loaded with the standard publisher trust check.
- A reduced, randomly initialized Qwen3.5 model passed four generation steps
  with three linear-attention layers and one full-attention layer on MPS.
  This is an architecture check, not the pretrained 9B model or a quality test.
- The real checkpoint is downloading. Full-model memory and latency are pending.

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

DeltaNet currently uses reference PyTorch implementations. The reduced test ran
on MPS, but this does not establish acceptable speed for the full model.

## Sources

- [Model](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Apple Silicon](https://huggingface.co/docs/transformers/main/apple_silicon)
- [Metal quantization](https://huggingface.co/docs/transformers/main/quantization/metal)
