# Completed four-method retrieval baseline

Historical result on the original human judgments. The active evaluation is
[pilot-qrels-v2](pilot-qrels-v2.md); the metrics below retain their original inputs.

The final supplemental annotation passed export and pool-binding validation. The common qrels now contain 200 judged query/article pairs. All previous 199 grades and all four top-10 rankings were preserved exactly. There are no remaining unjudged top-10 results.

Equal-weight RRF with c=60 and source depth 30 is below dense on all ten pilot questions. Keep it as an explicit experiment; dense remains the strongest measured quality reference.

| Metric (macro over 10 questions) | PubMed | BM25 | Dense | RRF |
| --- | ---: | ---: | ---: | ---: |
| ndcg@10 | 0.578 | 0.610 | 0.908 | 0.770 |
| precision@10 | 0.930 | 0.920 | 1.000 | 0.990 |
| mrr@10 | 1.000 | 0.950 | 1.000 | 1.000 |
| judged_coverage@10 | 1.000 | 1.000 | 1.000 | 1.000 |

| Question | Dense nDCG@10 | RRF nDCG@10 |
| --- | ---: | ---: |
| p01 | 0.990 | 0.849 |
| p02 | 0.820 | 0.690 |
| p03 | 1.000 | 0.853 |
| p04 | 0.958 | 0.901 |
| p05 | 0.948 | 0.831 |
| p06 | 0.998 | 0.831 |
| p07 | 0.957 | 0.687 |
| p08 | 0.776 | 0.759 |
| p09 | 0.747 | 0.704 |
| p10 | 0.891 | 0.592 |

## Decision

Fusion is not automatically an improvement. In this pool, equal-weight rank fusion lets the weaker lexical ranking pull down the stronger dense ordering. This is an observed property of the initial setting, not evidence that every hybrid configuration is inferior. No c, weights, or depth were tuned on these ten questions.

Future reranking should include dense alone as a comparator, alongside RRF. A Qwen or cross-encoder stage is useful only if its measured quality and latency justify the extra work relative to the strongest available baseline. Broader development and held-out evaluation remain necessary before choosing a final retrieval profile.

Precision counts partial and direct evidence as relevant. The graded nDCG metric distinguishes their usefulness. Fully judged top-10 lists support this pilot comparison but do not establish absolute candidate recall; 100 of the 300 candidate memberships are still unjudged. The pilot uses one annotator, with no independent agreement estimate.

## Artifacts

The [aggregate report](experiments/rrf-baseline.json) contains exact hashes, configuration, and per-query metrics. The [initial provisional run](experiments/rrf-retrieval.json) is retained for history. Local artifacts are:

- `data/evaluation/pilot-v1/rrf-supplement-human-001.jsonl`
- `data/evaluation/pilot-v1/human-expanded-rrf-001.jsonl` and its receipt
- `results/evaluation/pilot-v1-rrf-complete-001.json`

No retrieval or evaluation code changed during the final annotation review. Day 10 has a complete measured comparison; no further labels are needed for these four top-10 baselines.
