# First human-judged retrieval baseline

Historical result on the original human judgments. The active evaluation is
[pilot-qrels-v2](pilot-qrels-v2.md); the metrics below retain their original inputs.

The ten-query pilot is complete: one human annotator judged all 161 query/article pairs from the PubMed/BM25 top-10 union. Grades are 59 direct (2), 89 partial (1), and 13 irrelevant (0). The original p01 judgments are unchanged. No labels or ranking parameters were modified during this review.

BM25 has higher mean nDCG@10, but neither method wins consistently. Keep both baselines for the next retrieval experiments.

| Metric (macro over 10 queries) | PubMed | BM25 |
| --- | ---: | ---: |
| ndcg@10 | 0.670 | 0.708 |
| precision@10 | 0.930 | 0.920 |
| mrr@10 | 1.000 | 0.950 |
| judged_coverage@10 | 1.000 | 1.000 |

| Query | PubMed nDCG@10 | BM25 nDCG@10 |
| --- | ---: | ---: |
| p01 | 0.504 | 0.922 |
| p02 | 0.760 | 0.738 |
| p03 | 0.804 | 0.726 |
| p04 | 0.623 | 0.811 |
| p05 | 0.834 | 0.748 |
| p06 | 0.586 | 0.699 |
| p07 | 0.650 | 0.727 |
| p08 | 0.695 | 0.594 |
| p09 | 0.586 | 0.699 |
| p10 | 0.660 | 0.420 |

## Interpretation

BM25 wins on five queries and loses on five. Its mean advantage is 0.038 nDCG points. The large p01 gain drives most of that advantage: without p01, PubMed scores 0.689 and BM25 0.685. This diagnostic does not establish statistical significance or general superiority.

The largest BM25 regression is p10 (childhood anxiety): 0.420 versus 0.660. BM25 ranks a broad stress-management article (PMID 36096110, judged 0) first, while a pediatric anxiety guideline (PMID 32439401, judged 2) appears second in PubMed order and outside BM25 top-10. This is a concrete example to check with dense retrieval; it does not justify a special-case rule.

Binary Precision counts partial evidence as relevant and is high for both methods. The graded primary metric is more informative about direct answers. Fully judged top-10 results make this comparison usable, but only about 54% of candidate memberships were judged. Known-relevant candidate coverage is 1.0 by construction because labels came from these candidates; it cannot validate upstream retrieval recall.

These are ten development questions, one annotator, and title/abstract judgments. There is no independent agreement measurement or held-out result. The same pool will need expansion for dense/RRF top-10 before a fair comparison with those methods. Do not tune on future held-out questions.

## Artifacts

The [aggregate report](experiments/pilot-baseline.json) records exact input hashes, dataset IDs, per-query metrics, and the evaluation run ID. Human labels and full run outputs remain local:

- `data/evaluation/pilot-v1/human-final-001.jsonl`
- `data/evaluation/pilot-v1/human-final-001.jsonl.receipt.json`
- `results/evaluation/pilot-v1-human-final-001.json`

The validated export contains exactly the expected 161 pairs, no missing grades, and the matching pool ID. Existing evaluation code was used without changes. Day 6 now has a measured baseline; further annotation is not required before the next development step.
