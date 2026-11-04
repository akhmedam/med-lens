# Retrieval cost on the M3 Pro

Day 18 measures candidate-pool and reranking depth costs on the development
pilot. The primary pipeline decision remains Day 19. `ask.py` is unchanged.

## Protocol

- MacBook M3 Pro, 36 GB unified memory; MPS, float32, eager attention.
- Ten frozen pilot questions, original PubMed Best Match order, nested prefixes
  of 10, 20 and 30 articles. These are subsets of existing snapshots, not new
  PubMed searches or a measurement of a larger corpus.
- Dense uses BGE-base-en-v1.5, batch 16, exact cosine and maximum-window article
  scores. MedCPT uses its pinned revision, batch 4, all windows of each eligible
  article, maximum raw logit. No window boundaries, labels or runtime defaults
  changed. All pairs must fit 512 tokens without truncation.
- Dense → MedCPT depths: 10, 20 and 30 where the pool permits. The output remains
  top 10. Depth equal to pool scores the same windows as direct MedCPT and adds
  the Dense stage; exact article ties would follow different eligible orders.
- Three repetitions per question/configuration. One query/pair warm-up per model,
  excluded from results; there is no warm-up for every input shape. MedCPT
  traversal reverses on alternate repetitions. This is one local sequential
  session, not a randomized performance study.
- BGE and MedCPT run in separate processes. Frozen inputs are verified before
  timing. Every pool uses the matching subset of original windows/vectors;
  Dense cannot select an article outside its prefix. New passage vectors are
  checked against cached vectors and must preserve the complete article order.

## Timing boundaries

**Cached Dense:** query tokenization/encoding, cosine search and article
aggregation with a resident model and an in-memory index.

**Fresh Dense:** the same query encoding plus passage encoding from existing
window token IDs, construction of an in-memory vector view and article search.
It excludes fetching PubMed, parsing, building windows and writing a persistent
embedding artifact. This measures recomputing passage vectors, not the entire
cold ingestion workflow.

**Direct MedCPT:** pair tokenization, budget validation, forward passes,
aggregation and ranking report construction, with a resident model and validated
source data. It needs neither passage embeddings nor a query embedding.

**Cascade:** the sum of separately measured Dense and MedCPT stages for the same
question/repetition. These are composed estimates, not measured co-resident
end-to-end requests. They exclude model switching, loads and source validation.

Constructor wall times and source/index validation are recorded separately in
the JSON. Constructor times include asset verification and imports reached by
the constructor; they are single observations with potentially warm filesystem
caches, not cold application-start percentiles. Network/download time and Qwen
generation are outside every timing here.

The tables use the median across ten per-question medians. Min/max in the JSON
have the same query-median scope. These are not p95 latency estimates.

## Results

| Pool | Method | Dense depth | nDCG@10 | Known-pool Recall@10 | Judged top-10 coverage | Cached seconds | Recompute-vectors seconds | Pairs / 10 questions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | dense | — | 0.6851 | 0.4543 | 0.99 | 0.0079 | 0.5670 | 0 |
| 10 | medcpt | all | 0.6903 | 0.4543 | 0.99 | 0.4180 | 0.4180 | 279 |
| 10 | dense_medcpt | 10 | 0.6903 | 0.4543 | 0.99 | 0.4169 | 0.9828 | 279 |
| 20 | dense | — | 0.7917 | 0.4797 | 0.92 | 0.0079 | 1.2857 | 0 |
| 20 | medcpt | all | 0.8012 | 0.5036 | 0.93 | 0.8985 | 0.8985 | 618 |
| 20 | dense_medcpt | 10 | 0.7800 | 0.4797 | 0.92 | 0.4956 | 1.7523 | 319 |
| 20 | dense_medcpt | 20 | 0.8012 | 0.5036 | 0.93 | 0.9172 | 2.1936 | 618 |
| 30 | dense | — | 0.8637 | 0.5307 | 0.99 | 0.0089 | 2.0172 | 0 |
| 30 | medcpt | all | 0.8522 | 0.5387 | 0.99 | 1.3497 | 1.3497 | 975 |
| 30 | dense_medcpt | 10 | 0.8565 | 0.5307 | 0.99 | 0.4699 | 2.4773 | 350 |
| 30 | dense_medcpt | 20 | 0.8574 | 0.5387 | 0.99 | 0.9612 | 2.9841 | 639 |
| 30 | dense_medcpt | 30 | 0.8522 | 0.5387 | 0.99 | 1.3532 | 3.3779 | 975 |

PubMed Best Match remains at nDCG@10 0.6268 / known-pool Recall@10 0.4543
for every prefix. Its network/ingestion cost was not measured.

The pool contains 279 / 618 / 975 windows across the ten questions at
10 / 20 / 30 articles. Vector storage is 3,072 bytes per window (768 float32
components), about 0.82 / 1.81 / 2.86 MiB in total, excluding mapping metadata.

On the full 30-article pool:

- Cached Dense is about 9 ms. The depth-10 cascade costs about 0.470 s;
  direct MedCPT costs about 1.350 s.
- Recomputing passage vectors makes Dense about 2.017 s and the depth-10
  cascade about 2.477 s. Direct MedCPT avoids that encoding stage.
- Depth 20 roughly doubles cached cascade time to 0.961 s while changing
  pooled nDCG from 0.8565 to 0.8574; Dense is still higher at 0.8637.
- Depth 30 returns the same top ten as direct MedCPT for every question
  while adding Dense work. Direct/cascade rankings also match when depth
  equals the smaller pools.

The broad common subset with completely judged top tens contains only p03
and p07. It is too small to settle the entire depth curve. The earlier
[Day 17 comparison](medcpt-evaluation.md) retains its eight-query common subset
for the original full-pool methods. No labels or held-out questions changed.

| Isolated worker | Constructor seconds | Process peak RSS, GiB | Sampled MPS tensor max, GiB | Sampled MPS driver max, GiB |
| --- | --- | --- | --- | --- |
| bge | 2.910 | 0.921 | 0.428 | 2.011 |
| medcpt | 2.199 | 0.946 | 0.408 | 1.017 |

Both isolated retrieval workers fit comfortably in this laptop run. This
does not establish a memory budget for a combined retrieval/generation service.

The offline audit replayed all 270 MedCPT requests from 15,156 source-window
logit mappings, checked 90 Dense repetitions, and independently recalculated
480 per-query metric values. Full-pool Dense, depth-10 MedCPT and direct MedCPT
rankings match the frozen Day 16 artifact in every repetition. Torch,
Transformers, NumPy imports and socket creation were blocked during the audit.
This verifies replay and arithmetic, not the correctness of relevance labels.


## Quality limits

All configurations use the same full 30-candidate `pilot-qrels-v3` denominators,
including relevant judged papers outside a smaller prefix. PubMed top 10 is
unchanged across these prefixes. No new judgments were made to favor a depth.
The pool contains 210 pairs, 208 judged, with AI-assisted provenance; 90 of the
300 candidate pairs remain outside the judgment pool and two have explicit null
labels. Unjudged results contribute zero to pooled arithmetic, reduce reported
coverage and remain unjudged. Recall means known-pool recall.

The JSON reports top-10 coverage for every configuration and a common subset of
questions whose top tens are completely judged across all tested configurations.
Complete top-10 judgments do not imply complete candidate judgments. A depth
curve with newly retrieved unjudged articles cannot establish that those articles
are irrelevant. These are development results, not expert or held-out validation.

## Memory limits

Process RSS is a lifetime high-water mark for each isolated worker across all
pools and repetitions, including fresh passage encoding for BGE. It does not
measure cached Dense alone or the memory of a co-resident cascade.

MPS tensor/driver allocations are sampled after each forward batch by the existing
encoders. They are not instantaneous device peaks, and allocator retention can
carry across configurations. Unified memory means RSS and MPS observations must
not be added. Qwen was not loaded; these numbers do not establish the memory cost
of retrieval plus generation or a Medium collection.

## Reproduction

Install the existing embedding/reranking extras and prepare the same local pilot
snapshots, windows, vectors and pinned BGE/MedCPT assets described in
[Dense retrieval](dense-retrieval.md) and [MedCPT ranking](medcpt-ranking.md).
Run from the checkout, choosing unused output paths:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/benchmark_retrieval_cost.py \
  --output results/cost/new-run \
  --summary results/cost/new-run-public.json
```

The raw worker reports, all window logits and the exact measurement script remain
local. The compact [experiment report](experiments/retrieval-cost.json) preserves
runtime versions, source identities, code/report checksums, quality and cost
aggregates. Rebuild a detailed summary from completed workers without model
inference:

```sh
.venv/bin/python scripts/benchmark_retrieval_cost.py \
  --output results/cost/pilot-v1-001 --summarize-only
```

The measured worker function bodies must match the saved run source. The first
run's final aggregation used an incorrect qrels field name; its completed worker
outputs were preserved and the summary rebuilt after correcting the reader.
No model measurements were repeated or replaced for that fix.
