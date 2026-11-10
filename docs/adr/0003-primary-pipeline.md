# Primary search-to-answer pipeline

Status: accepted by the user on 2026-11-05.

## Decision

Use **Dense retrieval and article ranking**, followed by exact evidence selection
and **Qwen/Qwen3.5-9B answer generation**. Generating a sourced answer is a required
part of the product. Search-only output is a diagnostic mode and a useful partial
result when a model stage fails.

The primary request is:

```text
Question → BGE query embedding → Dense article ranking → exact source excerpts
         → Qwen3.5-9B draft → same-instance support review → validated answer/citations
```

Dense uses BAAI/bge-base-en-v1.5 at revision
`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`: normalized 768-dimensional final-layer
CLS embeddings, exact cosine similarity and maximum-window article scores. There
is no Qwen or MedCPT reranking stage in the selected pipeline.

Qwen uses revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, the existing
Transformers/MPS runtime and Metal 4-bit quantization, group size 64. One instance
handles the draft and support review sequentially after BGE is released. The
existing six-fragment / two-per-article / 4000-evidence-token limits and strict
citation/output gates remain in force.

## Reasons and limits

Dense offers the strongest measured quality/cost balance for repeated search
in a prepared collection. On the expanded pilot it scores 0.8637 nDCG@10 with a
resident cached query cost of about 0.009 s on the M3 Pro. The historical Qwen
reranker scores higher (0.8842), but adds a median 24.58 s per Dense top-10 list.
The MedCPT variants did not improve nDCG over Dense. These are small development
results with incomplete, AI-assisted judgments, not held-out confirmation.

The user selected Qwen9B as the generator. Its format and semantic reliability
remain Day 20–21 work: generation being required does not waive the validation
gates. Empty/insufficient evidence can produce an explicit abstention, malformed
answers are withheld, and completed search remains available after model errors.
Same-model support review is not independent verification.

Target: Air M3, 24 GiB. The [2026-11-07 recovery record](../experiments/air-recovery.json)
validates short inference and execution of the primary CLI with bounded Metal
conversion cache lifetimes; p01 produced a partial draft and p06 withheld an
overlong draft. This establishes
the measured runtime scope, not universal format or clinical reliability. Original
model revision, FP16/4-bit/group-64, budgets and semantic gates are unchanged.
Full-budget inference remains open; future model/quantization changes require
a separate explicit decision.

## Existing integration and continuation

The primary entry point is now `scripts/ask_live.py`: fetch current PubMed
candidates for each question and run the selected chain, with generation enabled
by default and no reranker. See [live requests](../live-search.md). This integrates
source acquisition without changing the chosen models or support-review policy.
For prepared-collection replay, `scripts/ask.py` still uses **`--generate` and
omits `--rerank`**. The planned interactive CLI will expose the same full workflow.

MedCPT, BM25/RRF and Qwen reranking remain historical comparison code until the
planned Day 27 cleanup. Do not remove their experiment records. Continue on the
Air with the staged [handoff checks](../handoff.md), then generation reliability
on Days 20–21 and held-out confirmation on Day 26.

The [current decision record](0003-primary-pipeline.json) is authoritative.
[Pipeline options](../pipeline-options.md) retain the measured trade-offs. The
Day 19 comparison packet and restore audit were captured before this choice;
their `pending-user` fields are historical snapshots and do not override this ADR.

## Native runtime refinement

The [native MLX runtime decision](0004-native-mlx-runtime.md) replaces the default
execution backend after the user requested performance optimization. It preserves
the selected retrieval/generation chain and pinned 9B checkpoint, while explicitly
recording arithmetic and stopping-policy differences. Independent answer quality
validation remains required.
