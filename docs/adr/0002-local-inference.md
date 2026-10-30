# Local inference

Status: accepted for experiments; reranking and cited draft generation measured

## Context

The laptop is an M3 Pro with 36 GB unified memory. We need both reranking and
answer synthesis, without loading two copies of the same model.

## Decision

Try Qwen/Qwen3.5-9B through Transformers and PyTorch on MPS, targeting 4-bit
linear weights. Check the installed Metal quantization path with a smoke test.
Do not assume documentation support proves this exact combination works.

Reuse one model instance sequentially. Reranking is listwise over ten papers;
invalid permutations preserve the input ranking. The initial RRF input is now
compared with dense top-10 as well, because dense outperformed RRF in the
[completed pilot](../rrf-baseline.md). Generation gets at most six evidence
fragments, two per paper, and 4000 evidence tokens, as starting experiment values.
The [evidence-context stage](../evidence-context.md) now implements these limits
and source mappings. Use system-assigned citation IDs and validate exact quotations. Show partial
answers or abstain when evidence is missing; present conflicting findings.

## Alternatives

MLX or llama.cpp may fit Apple Silicon better. A smaller model may run faster.
Revisit the runtime after a recorded failure, rather than changing it implicitly.

## Consequences

The smoke test measures compatibility and cost, not biomedical answer quality.
Keep model revision, package versions, prompt, output, and memory measurement
scope in its report. Do not label a sampled memory maximum a true device peak.

The [Day 1 report](../experiments/runtime-smoke.md) records the successful run,
two invalid reranking outputs, and the HTTP header workaround. The
[reranking pilot](../qwen-reranking-baseline.md) now measures real abstract inputs,
latency and ranking quality. The [answer pilot](../cited-answers.md) uses the same
runtime for sequential draft and support-review calls, with 768 and 512 output
tokens respectively. Sentence IDs resolve to exact source slices. The pilot
uses less than the full 4000-token evidence budget; testing at that limit remains
open. Same-model review has missed population and qualifier errors, so generated
answers remain research drafts requiring independent review.
