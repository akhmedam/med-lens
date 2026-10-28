# Qwen reranking on the pilot

Qwen improves mean nDCG@10 when reordering either frozen source top-10. Dense
plus Qwen is the strongest measured variant, but the gain over dense alone is
small and costs about 26 seconds per question on the M3 Pro. Keep this stage
opt-in while expanding evaluation; dense alone remains the lower-latency reference.

| Method | nDCG@10 | Precision@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| Dense | 0.908470 | 1.00 | 1.00 |
| Dense + Qwen | **0.926376** | 1.00 | 1.00 |
| RRF | 0.769654 | 0.99 | 1.00 |
| RRF + Qwen | 0.826028 | 0.99 | 1.00 |

All four top-10 lists are fully judged. Source rankings, datasets and all 200
human judgments are unchanged from the completed retrieval baseline. No new
annotation was needed. The model saw no human labels or retrieval scores.

## Per-question comparison

| Question | Dense | Dense + Qwen | RRF | RRF + Qwen |
| --- | ---: | ---: | ---: | ---: |
| p01 | 0.990403 | 0.987960 | 0.849294 | 0.858892 |
| p02 | 0.819928 | 0.905536 | 0.690296 | 0.788470 |
| p03 | 1.000000 | 1.000000 | 0.852538 | 0.861151 |
| p04 | 0.957586 | 0.955831 | 0.900824 | 0.905321 |
| p05 | 0.947734 | 0.947734 | 0.831136 | 0.854656 |
| p06 | 0.998167 | 0.993217 | 0.831023 | 0.839477 |
| p07 | 0.957013 | 0.957702 | 0.686872 | 0.833658 |
| p08 | 0.776115 | 0.776115 | 0.758839 | 0.769091 |
| p09 | 0.747200 | 0.836863 | 0.703661 | 0.802321 |
| p10 | 0.890553 | 0.902799 | 0.592062 | 0.747247 |

Dense reranking improves four questions, ties three and worsens three. Its mean
gain is 0.017906, concentrated mainly in p02 and p09. RRF reranking improves all
ten questions, with a mean gain of 0.056374, yet remains below unreranked dense.
Improving a weaker baseline is insufficient to choose the final pipeline.

A complete permutation cannot change Precision@10 or Recall@10 when the input
contains ten papers. The completed run verifies that membership, precision,
pooled recall and judged coverage are preserved for every call. nDCG measures
whether direct evidence moves above partial evidence. Binary MRR was already
1.0 and stays there.

## Runtime and validation

The fixed [listwise-v1 policy](llm-reranking.md) uses the pinned Qwen3.5-9B revision,
4-bit Metal linear layers, greedy decoding and no thinking. Twenty sequential
calls reused one model instance. All twenty returned valid complete permutations;
no fallback, repair or retry was needed during the completed quality run.

- Generation including prefill: median 25.73 seconds, range 18.87–33.34 seconds.
- Sum across twenty calls: 513.53 seconds; loading/tokenization/report writing excluded.
- Model/tokenizer loading: 21.41 seconds, using cached weights.
- Process peak RSS: 14.44 GiB, including model loading. This is not a GPU or total
  unified-memory peak and should not be added to overlapping device counters.
- Prompts: 2796–5495 Qwen tokens. 10 of 200 document presentations were shortened
  to the 768-token per-paper limit; repeat appearances count separately.
- 114 offline unit tests pass, including malformed output, fallback, input limits,
  label-free prompt construction and baseline hash/metric binding. A separate
  real-tokenizer audit matched prompt counts to tensor inputs for all twenty cases.

Two setup attempts stopped before any quality result: one exposed chat-template
return-format handling, and one could not resolve the Metal kernel in offline
mode. Both failed reports remain local. The successful run uses the corrected
explicit token-list format and the standard online kernel resolution/trust check;
model/tokenizer weights still load locally. No prompt tuning followed quality results.

## Limits and decision

This is the same ten-question development pilot, with one annotator and pooled
judgments. There is no held-out claim, statistical significance estimate or
independent annotation agreement. The reranker only reorders ten candidates,
uses source order as input, and has not been tested for position sensitivity or
repeat-run ranking stability. Truncated abstracts may omit useful conclusions.

Keep dense + Qwen available as an explicit quality experiment. Do not enable it
implicitly for every search or select a final architecture from this pilot.
Broader development and held-out checks must establish whether the small gain
justifies the latency. The previously agreed specialized cross-encoder remains
an alternative if the LLM gain does not hold or its runtime is unsuitable; it
has not been implemented or measured here.

The [aggregate report](experiments/qwen-reranking.json) records exact source/report
hashes, environment, configuration, per-question metrics and timings. Full prompts,
article text, raw model outputs and individual human labels stay local in
`results/evaluation/pilot-v1-qwen-rerank-003.json`.
