# Pipeline decision: Dense and Qwen9B

Status: accepted by the user on 2026-11-05: **Dense → evidence context →
Qwen3.5-9B answer generation and support review**, with no additional reranker.
Generation is required in the primary product workflow; search-only is diagnostic.
The [accepted ADR](adr/0003-primary-pipeline.md) records the decision. Target:
MacBook Air M3, 24 GB; all timings below come from the M3 Pro, 36 GB.

## Retrieval options

The same expanded v3 development pool is used throughout: ten questions, 210
query/article pairs, 208 judged. Every snapshot contains 30 candidate articles.

| Retrieval | nDCG@10 | Known-pool Recall@10 | Cached seconds | Recompute-vectors seconds | Current request integration |
| --- | --- | --- | --- | --- | --- |
| Dense | 0.8637 | 0.5307 | 0.009 | 2.017 | `ask.py`, including excerpts and optional draft |
| Dense → MedCPT, depth 10 | 0.8565 | 0.5307 | 0.470 | 2.477 | Standalone ranking; answer integration pending |
| Dense → MedCPT, depth 20 | 0.8574 | 0.5387 | 0.961 | 2.984 | Benchmark; parameterized request integration pending |
| Direct MedCPT | 0.8522 | 0.5387 | 1.350 | 1.350 | Standalone ranking; answer integration pending |

Times exclude loads, PubMed fetching, parsing/windowing and answer generation.
Cascade totals sum separately measured stages. Fresh-vector timing starts with
already prepared windows; it is not full cold ingestion. See the
[cost protocol](retrieval-cost.md) for timing and memory scopes.

Dense is the recommended base for repeated search in a prepared collection: the
best measured nDCG among these four options and a low query cost. Direct MedCPT
is a reasonable alternative for one-off fresh candidate lists, where avoiding
passage embedding construction can matter. It slightly increases known-pool
Recall here. The depth-10 cascade does not improve nDCG over Dense; depth 20
nearly doubles its cost for a small pooled-metric change.

A historical Dense → Qwen listwise rerank scores **0.8842 nDCG@10** after rescoring
on the same v3 labels. Its Dense-input calls took a median **24.58 s**, excluding
model loading (25.73 s combines Dense and RRF input calls). It is the largest
observed retrieval score and should not be hidden. It is an expensive historical
experiment with a different input protocol, not a new Day 18 measurement or
proof of a reliable gain on unseen questions. The accepted plan treats Qwen
reranking as historical, retaining Qwen for generation/review. Choosing to keep
Qwen reranking as the primary method would explicitly revise that plan.

PubMed/BM25/RRF controls are documented in the
[expanded comparison](medcpt-evaluation.md). Alternative implementations and assets
remain available until the planned Day 27 cleanup.

## Answer modes

The primary workflow includes Qwen generation. Existing operational modes are:

1. **Articles and exact source excerpts.** Existing Dense `ask.py` without
   `--generate`. No Qwen weights load; the cached tokenizer is used for evidence
   budgeting. This diagnostic mode is the first staged check on the Air.
2. **Selected primary workflow: cited Qwen draft.** The same evidence enters Qwen3.5-9B, followed by
   support review using the same model instance. Quotes, IDs and output format
   are checked; invalid output is withheld. An accepted draft is still subject
   to independent review.

The Day 13 answer pilot used Dense+Qwen-reranked contexts: four of ten questions
produced visible drafts (nine accepted claims); six failed strict output gates.
This is not a controlled answer-quality comparison across the retrieval methods
above. The Day 14 full request still omitted an older-adult qualifier, and the
same-model reviewer accepted it. Higher nDCG alone cannot choose a generator or
establish that its conclusions preserve the source population and comparisons.

The accepted pipeline is Dense plus exact evidence and Qwen9B generation.
Days 20–21 address format and semantic failures; output remains an experimental
research draft until those release checks are met.

## Air M3 checks before enabling drafts

The small retrieval workers each peaked below 1 GiB process RSS in isolated
Day 18 runs; sampled MPS driver allocations reached about 2.01 GiB for BGE and
1.02 GiB for MedCPT. Those observations do not measure combined models or the
Air's sustained performance.

A historical Qwen smoke run reached **15.03 GB process peak RSS**, with sampled
MPS tensor allocation **8.16 GB** and driver allocation **28.59 GB**. These are
overlapping observations from unified memory and must not be added. The driver
sample exceeds the Air's physical 24 GiB capacity; it does not by itself prove an
out-of-memory outcome there. Qwen's load/conversion transient and full-budget
request must be measured on the Air before calling that configuration supported.
The approximately 18 GB on-disk weight cache is not its quantized runtime footprint.

Use the staged checks in [the handoff guide](handoff.md). Do not raise allocator
limits, silently reduce evidence budgets, swap models or enable CPU fallback to
make a nominal success. A smaller model/runtime is an explicit new experiment
if the existing Qwen path does not fit or is too slow.

## Continue with the accepted pipeline

Run `ask.py --generate` without `--rerank`. This already connects Dense, context
selection, Qwen drafting and support review. Verify the staged Air checks, then
continue Days 20–21. A future MedCPT integration would require explicit raw-logit
context support; it is outside the selected primary path.

The [machine-readable accepted decision](adr/0003-primary-pipeline.json) records
the current choice. The earlier [comparison packet](experiments/pipeline-options.json)
and restore audit remain immutable pre-decision evidence; their `pending-user`
fields describe when those snapshots were captured. This command rebuilds that
pre-decision evidence, not the current accepted ADR:

```sh
.venv/bin/python scripts/prepare_pipeline_decision.py \
  --output results/decision/new-options.json
```

The held-out questions remain reserved for Day 26. The selected pipeline is
implemented. [Air restoration](experiments/air-recovery.json) now records short
inference and primary-CLI execution with bounded conversion cache lifetimes.
Full-budget inference and generation quality work remain open.
