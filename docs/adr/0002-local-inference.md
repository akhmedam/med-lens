# Local inference

Status: selected; runtime verification pending

## Context

The laptop is an M3 Pro with 36 GB unified memory. We need both reranking and
answer synthesis, without loading two copies of the same model.

## Decision

Try Qwen/Qwen3.5-9B through Transformers and PyTorch on MPS, targeting 4-bit
linear weights. Check the installed Metal quantization path with a smoke test.
Do not assume documentation support proves this exact combination works.

Reuse one model instance sequentially. Reranking is listwise over RRF's top-10;
invalid permutations fall back to RRF. Generation gets at most six evidence
fragments, two per paper, and 4000 evidence tokens, as starting experiment values.
Use system-assigned citation IDs and validate exact quotations. Show partial
answers or abstain when evidence is missing; present conflicting findings.

## Alternatives

MLX or llama.cpp may fit Apple Silicon better. A smaller model may run faster.
Revisit the runtime after a recorded failure, rather than changing it implicitly.

## Consequences

The smoke test measures compatibility and cost, not biomedical answer quality.
Keep model revision, package versions, prompt, output, and memory measurement
scope in its report. Do not label a sampled memory maximum a true device peak.
