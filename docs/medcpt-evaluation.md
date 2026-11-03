# MedCPT pilot evaluation

MedCPT did not improve the primary nDCG@10 metric on this ten-question pilot.
Dense → MedCPT was slightly below Dense, and direct MedCPT was below both.
Direct MedCPT retrieved slightly more known relevant papers, but ordered the
results less usefully under the current graded rubric. This is a development
result with AI judgments, not an independent or clinical validation.

## Shared inputs and judgments

The [comparison profile](../benchmarks/pilot-v1/medcpt-evaluation.json) pins the
Day 16 rankings, original v2 profile and supplemental annotation artifacts.
The source snapshots and all method rankings are unchanged. Fresh Dense order
matches the historical Dense order on all ten questions, so the saved Qwen
permutations can be rescored alongside the new methods.

The expanded qrels v3 pool has 210 query/article pairs: 208 judged and two null.
All 200 v2 entries, including its two unresolved entries, are preserved. Ten new
pairs from the new top tens were assessed by Codex from their complete frozen
titles and primary abstracts: five direct and five partial judgments. Each has
a source/version binding, confidence and rationale. Method names, scores and
ranks were absent from the supplemental item file, but prior conversation context
included some rankings. This is not independent blinded assessment or expert
human ground truth. The prior implicit-adult-age convention remains a limitation.

The historical `benchmarks/pilot-v1/evaluation.json` stays on v2 for existing
artifact replay. The new profile is a separate expanded-pool comparison; do not
pass it to `rescore_pilot.py`. Old v2 metric values and new v3 values should not
be compared as if they used the same judgment denominator. Every method below
has been rescored on the same expanded pool.

## Results

Macro averages over all ten questions, cutoff ten:

| Method | nDCG@10 | Known-pool Recall@10 | MRR@10 | Judged coverage@10 |
| --- | ---: | ---: | ---: | ---: |
| PubMed Best Match | 0.6268 | 0.4543 | 0.9500 | 0.9900 |
| Dense | 0.8637 | 0.5307 | 1.0000 | 0.9900 |
| Dense → MedCPT | 0.8565 | 0.5307 | 1.0000 | 0.9900 |
| Direct MedCPT | 0.8522 | 0.5387 | 1.0000 | 0.9900 |
| Dense → Qwen, historical reference | 0.8842 | 0.5307 | 1.0000 | 0.9900 |

BM25, RRF and RRF → Qwen are also rescored in the full
[aggregate report](experiments/medcpt-evaluation.json). Qwen uses its historical
prompt/input policy; this compares pipelines, not isolated model architectures.
No new retrieval or model inference was run for these metrics.

Recall counts grades 1 and 2 as relevant; nDCG weights grade 2 more heavily.
Dense → MedCPT reorders exactly the same ten articles, so its Recall/Precision
cannot improve over Dense in this experiment. Direct MedCPT can change membership
and gains a little Recall, but that does not guarantee better graded ordering.

Coverage remains incomplete: PMID 26646303 for p01 and PMID 33423271 for p04
have no primary abstracts and stay unjudged. Each primary method has one unknown
slot across its hundred returned results. The report is explicitly
`provisional-incomplete-judgments`. Only 208 of the 300 candidate query/article
pairs are judged; these metrics do not measure recall over all relevant PubMed
literature or a fully labeled candidate corpus.

## Uncertainty checks

The common subset p02, p03, p05, p06, p07, p08, p09 and p10 has fully judged top
tens for all four primary methods. Candidate pools are still only partly judged.
On these eight questions:

| Method | nDCG@10 | Known-pool Recall@10 |
| --- | ---: | ---: |
| PubMed | 0.6561 | 0.4685 |
| Dense | 0.8516 | 0.5138 |
| Dense → MedCPT | 0.8437 | 0.5138 |
| Direct MedCPT | 0.8410 | 0.5317 |
| Dense → Qwen | 0.8795 | 0.5138 |

Assigning each of the two explicit nulls every possible grade from 0 through 2
produces nine exact scenarios. IDCG and Recall denominators are recomputed for
every scenario, rather than holding the normalization fixed:

| Paired mean nDCG@10 difference | Minimum | Maximum |
| --- | ---: | ---: |
| Dense → MedCPT minus Dense | -0.00714 | -0.00480 |
| Direct MedCPT minus Dense | -0.01144 | -0.00910 |
| Direct MedCPT minus Dense → MedCPT | -0.00430 | -0.00430 |
| Direct MedCPT minus Dense → Qwen | -0.03195 | -0.02769 |

These bounds vary only the two explicitly unresolved pairs. They are not
confidence intervals and do not cover wrong AI labels or the ninety unpooled
candidate pairs. With the ten fixed development questions, no statistical
superiority or broad generalization claim is made.

A second check lowers both new medium-confidence direct judgments to partial:
one has mixed panic/agoraphobia diagnoses, the other lacks explicit adult ages.
All older judgments stay fixed. nDCG becomes Dense 0.8673, Dense → MedCPT 0.8599,
direct MedCPT 0.8466 and Dense → Qwen 0.8878. The direction of the comparison is
unchanged. This is a limited labeling sensitivity check, not independent review.

Dense → MedCPT wins on two questions, ties on one and loses on seven by nDCG.
Direct MedCPT wins against Dense on one question and loses on nine. Individual
query results and paired deltas remain in the aggregate report.

## Source-based error analysis

The [case record](experiments/medcpt-error-analysis.json) documents four
illustrative cases, with detailed source references kept locally:

- **p02, improvement:** MedCPT demotes adolescent-only evidence in the adult iCBT
  question. Direct scoring also retrieves a useful adult meta-analysis outside
  Dense's ten candidates. nDCG rises from 0.6410 to 0.8142/0.8326.
- **p07, loss:** broad GAD/panic and mixed-anxiety treatment summaries move above
  evidence more directly addressing adult GAD and SSRIs under the frozen rubric.
  Dense → MedCPT drops from 0.7955 to 0.6261.
- **p05, loss:** direct scoring promotes a mobile-CBT review with pooled ages
  13–25 without separate adolescent estimates. Its maximum-scoring excerpt is
  a methods section. Direct nDCG falls from 0.9576 to 0.8633.
- **p09, loss:** a methodological review moves above direct symptom-outcome
  reports; thematic matching does not always preserve answer-specific usefulness.

These are observations from source inspection, not a causal ablation of max
pooling, an exhaustive error rate or clinical assessment. No judgments were
changed after inspecting metric outcomes, and no model/window/prompt tuning
was performed in this evaluation.

## Replay

Requires the ignored frozen source datasets/windows, Day 16 ranking artifact,
historical replay inputs and qrels/provenance files named by the profile:

```sh
python scripts/evaluate_medcpt.py \
  --output results/evaluation/medcpt-example.json \
  --summary results/evaluation/medcpt-example-summary.json
```

Choose new paths. The runner validates input hashes, preserved v2 grades,
supplement source text/version/rationale bindings, candidate identities, the
Dense baseline and MedCPT maximum-window replay. It rejects changed rankings,
missing source-map fields, invalid review labels and inconsistent counts.

The canonical local output is `results/evaluation/pilot-v1-medcpt-eval-003.json`.
The [audit](experiments/medcpt-evaluation-audit.json) independently recalculates
320 metric values and checks all nine null-label scenarios. The complete report
replays identically with network access and imports of Torch, Transformers and
NumPy blocked. No model weights are needed for evaluation. All 188 tests pass.
These checks validate arithmetic and provenance, not the correctness of AI labels.

## Next decision

Keep the measured alternatives until the Day 18 cost/scale comparison and Day 19
selection. There is no quality evidence here for replacing Dense with MedCPT.
The strongest historical reference remains Dense → Qwen, with its documented
latency cost. More questions, better independent judgments and the untouched
held-out split are needed before generalizing beyond this pilot.
