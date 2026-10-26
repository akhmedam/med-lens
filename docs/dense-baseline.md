# Dense baseline on the expanded human-judged pool

The 38 supplemental cards are complete: 31 direct (2), seven partial (1), no missing grades. Export and pool binding validation passed. The merge retained all 161 earlier labels exactly, giving 199 judged query/article pairs. The three top-10 rankings are unchanged from the provisional run. No ranking parameters or human labels were modified during this review.

On this pilot, dense retrieval has higher nDCG@10 than PubMed and BM25 on all ten questions. This supports continuing with the planned hybrid retrieval experiment; it does not establish general superiority outside the pilot.

| Metric (macro over 10 questions) | PubMed | BM25 | Dense |
| --- | ---: | ---: | ---: |
| ndcg@10 | 0.578 | 0.610 | 0.908 |
| precision@10 | 0.930 | 0.920 | 1.000 |
| recall@10 | 0.499 | 0.500 | 0.542 |
| mrr@10 | 1.000 | 0.950 | 1.000 |
| judged_coverage@10 | 1.000 | 1.000 | 1.000 |

| Question | PubMed nDCG@10 | BM25 nDCG@10 | Dense nDCG@10 |
| --- | ---: | ---: | ---: |
| p01 | 0.398 | 0.728 | 0.990 |
| p02 | 0.688 | 0.668 | 0.820 |
| p03 | 0.804 | 0.726 | 1.000 |
| p04 | 0.623 | 0.811 | 0.958 |
| p05 | 0.834 | 0.748 | 0.948 |
| p06 | 0.469 | 0.559 | 0.998 |
| p07 | 0.381 | 0.427 | 0.957 |
| p08 | 0.586 | 0.501 | 0.776 |
| p09 | 0.549 | 0.654 | 0.747 |
| p10 | 0.446 | 0.283 | 0.891 |

## Interpretation and limits

Top-10 judged coverage is now 100% for each method. Precision includes both partial and direct evidence; nDCG rewards direct answers more strongly. Dense Precision@10 is 1.0 on this annotation set. The largest previous BM25 regression, childhood anxiety (p10), now scores 0.891 for dense versus 0.283 for BM25 and 0.446 for PubMed.

PubMed/BM25 nDCG values fell relative to the old 161-pair report because the expanded pool contains additional highly relevant articles. Ideal DCG increased even though rankings and old labels did not change. Recall denominators also expanded. Compare methods within one qrels version; differences across pool versions do not indicate ranking regressions.

This remains a ten-question development pilot with a single annotator. The supplemental batch was judged later, and inter-annotator agreement was not measured. Fully judged top-10 lists do not establish absolute recall: only 199 of the 300 candidate memberships are judged, and none of the methods can retrieve outside the saved candidates. Known-relevant candidate coverage remains 1.0 by pool construction.

No additional annotation is needed for this three-method baseline. RRF may introduce other candidates into its top-10; expand the pool if that happens before claiming a fair four-method comparison. Keep these query groups out of the held-out set.

## Artifacts

The [aggregate report](experiments/dense-baseline.json) stores hashes, per-query metrics, and the evaluation run ID. Raw annotations and full outputs remain local:

- `data/evaluation/pilot-v1/dense-supplement-human-001.jsonl`
- `data/evaluation/pilot-v1/human-expanded-001.jsonl` and its provenance receipt
- `results/evaluation/pilot-v1-dense-expanded-001.json`

Validation used the existing export/merge/evaluation implementation without code changes. All 38 exported grades matched the expected supplemental pairs, all 161 prior grades were preserved, all 199 grades are present, and every returned top-10 article is judged.
