# Broader-topic exploratory quality assessment

On 2026-11-12, the six frozen biomedical development questions were assessed
within one working session. Dense ranked the judged pool better than PubMed Best
Match, but the current 4B answer/review pipeline did not produce a fully usable
answer on this strict source-bound assessment: **two answers were partial and
four required correction**. This is an AI assessment, not independent human or
clinical validation. The 4B development default remains unchanged.

The subsequent [scoped-findings repair experiment](biomedical-quality-improvement.md)
is reported separately. It adds an opt-in quotation format; this baseline and its
original judgments are preserved.

The [machine-readable record](experiments/biomedical-quality.json) contains metric
denominators, per-case judgments, model/prompt/source identities, timings and
artifact hashes. Frozen titles/abstracts and original labels remain in the local
`results/biomedical-dev-v1` tree. Raw prompts/outputs and review quotes are in
`results/biomedical-dev-v1/quality-20261112-v1`.

## Protocol and provenance

The original source plan and blank human packet remain immutable. After source
preparation, the user requested a separate assistant annotation of all 84
question/article pairs, then a one-day quality assessment. The separate evaluation
protocol was frozen before answer generation (SHA256
`495ed18b8a135d7c48556a0847766acb5ea258992db1e34f6103ae005571451e`).
All new judgments explicitly have `human_authored=false`; genuine historical
human labels are unchanged. The independent-human gates in ADR 0006 are not
claimed satisfied by this exploratory route.

Retrieval uses the completed source-bound AI labels. Answer assessment is a
separate review of all 18 raw claims and 17 visible claims against the selected
excerpts and complete frozen source abstracts, without external medical facts.
There were two disjoint primary batches, an isolated second-pass reviewer and
explicit root adjudication. Seven claims had an axis disagreement and one answer
had a category disagreement before adjudication. Verdicts were hidden in the
review packets; complete primary blinding is not claimed because one primary
inherited parent context. Reviewer agreement is not accuracy.

Factual support, requested population/comparison/outcome alignment, preservation
of qualifiers and answer completeness are separate judgments. A correct statement
about a composite endpoint can still miss the requested component. An accurate
statement about another medicine is not an answer to the requested intervention.
Unknown scope is retained as uncertainty, not counted as a fabricated fact.
The held-out questions were not read or used.

## Retrieval

The pool is the union of up to ten PubMed Best Match results and ten Dense
results from each identical snapshot. Every returned result is judged, but only
84 of the 246 candidate pairs are judged overall. Other candidates remain
unjudged. Gains for nDCG are `2**grade - 1`; the ideal ranking is pool-limited.

| Macro metric | PubMed Best Match | Dense | Defined questions |
| --- | ---: | ---: | ---: |
| Graded nDCG@10 | 0.5376 | 0.7888 | 5/6 |
| P@10, useful relevance ≥1 | 0.5500 | 0.7667 | 6/6 |
| Pool Recall@10, relevance ≥1 | 0.5384 | 0.7674 | 5/6 |
| P@10, direct relevance =2 | 0.1667 | 0.3000 | 6/6 |
| Pool Recall@10, direct relevance =2 | 0.5133 | 0.7933 | 5/6 |

Precision uses a fixed denominator of ten, including short lists. Pool recall is
not recall over the candidate acquisition stage or all PubMed literature.
For `bm06`, all four candidates have relevance zero, so nDCG and recall are
undefined, not zero; that case is excluded from those macro denominators.
No significance or generalization claim follows from these six development cases.

## Answer quality

| Case | AI assessment of visible answer | Main limitation |
| --- | --- | --- |
| bm01, heart failure | Partial | Composite hospitalization/death results are repeated instead of answering the separately requested hospitalization endpoint. |
| bm02, diabetes | Partial | A qualitative direct comparison is mixed with a numerical add-on/dose-adjustment comparison. |
| bm03, UTI diagnosis | Requires correction | The answer mixes culture, consensus and symptom-plus-culture reference definitions. |
| bm04, MCI prognosis | Requires correction | The core APOE association is supported, but one AD-framed numerical result comes from a dementia-unspecified cohort. |
| bm05, NLRP3 mechanism | Requires correction | Mouse/model boundaries and the distinction between human IL-1 evidence and the proposed upstream NLRP3 mechanism are lost. |
| bm06, pediatric otitis | Requires correction | The answer discusses analgesics and other medicines instead of antibiotics versus placebo; it should withhold the requested conclusion. |

Of 17 visible claims, 14 were factually supported by the frozen abstracts, one
misstated a material reference definition, and two had unresolved factual support.
Only five also passed every applicable requested-scope and qualifier axis.
These counts do not mean that the other twelve are twelve fabricated medical
facts: several are accurate but incomplete, indirect or outside the question.

The same-model reviewer rejected one raw claim, correctly excluding a numerical
mix-up in the diabetes case. It still accepted material scope/qualifier failures.
All twelve draft/review calls completed without output-cap or contract failure;
exact citation checks passed. Format and citation validity did not establish
scientific correctness. No fully usable answer or appropriate abstention was
recorded under this protocol; this small AI-assessed sample is not a clinical
accuracy estimate.

## Runtime and targeted query diagnostic

Each fixed-source replay took **33.87–38.79 s**, mean 36.75 s. This includes source
checks, query encoding, fresh model loading, generation/review and cleanup, but
excludes fresh PubMed acquisition and building passage embeddings. All six Dense
rankings, selected sources and evidence payloads matched their original runs;
all calls were fresh and workers were closed and reaped. The runtime was the
pinned four-bit Qwen3.5-4B MLX artifact with `cited-answer-v8` / `axes-first-v2`.
No paired speedup against a historical workflow is established.

The default otitis query retains `after two to three days`; PubMed translates
these words into mandatory search terms. Three ESearch probes found:

| Query variant | Candidates |
| --- | ---: |
| Original lexical query | 4 |
| Same query without the temporal phrase | 25 |
| `acute otitis media children antibiotics placebo` | 169 |

Counts alone do not measure relevance. One separate live diagnostic used the
existing explicit `--pubmed-query` override without the temporal phrase, retaining
the complete original question for Dense and Qwen. It fetched 25 articles and
completed in **44.75 s of request time**. The new answer gave accurate 2–7-day and
first-two-day context, but still missed the exact 2–3-day outcome. That outcome is
present in full source abstracts and was omitted by the selected excerpts.
The generated duration-comparison claim was rejected by support review.

This is a separate fresh-source diagnostic, not an extra observation in the six
baseline cases or a paired quality improvement. No old labels were transferred,
and the default query/evidence/model policy was not changed.

## Remaining defects and development decision

1. **P0 — requested-scope validation.** Reject or explicitly qualify wrong
   interventions, reference standards, experimental systems and endpoint
   substitutions. Same-model review currently misses these errors.
2. **P1 — evidence coverage.** Include the methods/qualifiers and specific outcomes
   needed by the question; six high-similarity excerpts can omit them even when
   the full retrieved abstract contains the answer.
3. **P1 — query construction.** Avoid overconstraining candidate acquisition with
   literal temporal prose while preserving the full original question downstream.
   The existing manual override is a diagnostic workaround, not an automatic fix.

The one-day assessment is complete. API and interactive CLI work may proceed with
explicit research-draft status. Material semantic-support errors remain release
blockers; the evaluation does not approve autonomous biomedical answers or settle
whether another model would solve these problems.

## Reproduction and verification

With the existing local frozen artifacts:

```sh
.venv/bin/python scripts/evaluate_biomedical_retrieval.py \
  --sources results/biomedical-dev-v1/sources-20261111-v2 \
  --annotation-package results/biomedical-dev-v1/manual-annotation-20261111-v1 \
  --assistant-annotations results/biomedical-dev-v1/assistant-annotation-20261111-v1 \
  --output results/biomedical-dev-v1/retrieval-new-run.json

.venv/bin/python results/biomedical-dev-v1/quality-20261112-v1/summarize_answers.py \
  --artifacts results/biomedical-dev-v1/quality-20261112-v1 \
  --judgments results/biomedical-dev-v1/quality-20261112-v1/answer-judgments.json \
  --output results/biomedical-dev-v1/answers-new-run.json
```

Outputs must be fresh; no reference or prior result is overwritten. The retrieval
scorer fails on changed source versions, incomplete labels or mismatched artifact
bindings. It accepts only the separate explicitly assistant-origin path; the human
annotation importer still rejects AI exports.

The full suite ran 408 tests: **401 passed, seven optional native-grammar tests
skipped in the main environment**. Seven new fictional tests cover undefined
relevance denominators, short lists, missing/unknown labels and changed source
bindings. The actual twelve native model calls above supply separate runtime
execution evidence. Source/package/annotation integrity checks passed.
