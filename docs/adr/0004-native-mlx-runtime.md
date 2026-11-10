# Use native MLX for Qwen inference on Apple Silicon

Status: accepted for the development runtime, subject to independent answer
quality validation. The user requested latency optimization and authorized
architecture/runtime/model experiments on 2026-11-10.

The 9B model/default choice below is superseded by
[ADR 0005](0005-qwen4b-development-default.md). The isolated native runtime
architecture remains selected; model-specific measurements here are historical.

## Problem and decision

Live PubMed acquisition costs a few seconds; Qwen load/quantization and two model
calls dominate the original request latency. Transformers/MPS executes the
Qwen3.5 recurrent-attention path without its optional optimized FLA/causal-conv
extensions. This is not evidence of implicit CPU fallback. Native MLX-LM 0.32.2
provides an Apple GPU implementation of the model and prepared quantized loading.

Keep the pinned Qwen3.5-9B checkpoint, Dense ranking, evidence selection and v6/v4
draft/review contracts. Run the native model in a separate pinned environment and
subprocess, loaded once for both calls. Select the closer quantization coverage:
4-bit/group-64 linear layers with FP16 embeddings and output head. Native A_log
remains FP32; no bitwise or semantic equivalence with the original runtime is
claimed. The all-layer/embedding 4-bit variant is a separate, unselected experiment.

The former Transformers/Metal backend remains explicit via `--engine transformers`.
There is no automatic fallback. Both backends explicitly stop on source EOS and
chat end. Finite-score, token-budget, complete-JSON, exact-citation and support
review gates remain enabled. Invalid drafts/reviews are withheld.

## Evidence and limits

See [runtime setup and contracts](../native-mlx.md) and the
[performance record](../experiments/native-mlx.json). Fixed-source pipeline replay
and fixed-input timing distinguish runtime work from PubMed result changes. Fresh
live smoke requests exercise the integrated path and vector reuse. Output lengths,
automatic statuses and precision changes are disclosed; runtime timing does not
judge clinical support or relevance. Single sequential observations are not a
latency distribution or proof of a uniform speedup.

Full 4-bit embedding/head coverage reduces memory but increased format failures
in the measured development contexts. It is not selected merely for a shorter
latency. A smaller model is an experimental tradeoff until independently assessed;
existing quality labels are not transferred to changed sources or model outputs.

## Consequences

Native setup adds a separate environment and an offline prepared model artifact.
The parent keeps BGE and the original HF tokenizer; exact IDs cross a bounded
protocol. The pipeline closes/reaps its worker on every exit path. Fresh per-call
KV/recurrent state prevents reuse across different prompts. A manifest binds model
provenance, conversion policy, packages and file hashes. Request identity includes
the chosen backend/manifest and actual prompt token hashes.

This improves local execution cost. Remaining semantic-support errors and manual
quality evaluation still gate the release. Do not infer answer-quality gains from
format acceptance or same-model review.
