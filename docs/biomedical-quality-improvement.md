# Scoped source findings: development quality repair

The unchanged-source broader-topic baseline produced no fully usable answer on six development questions. The new **experimental opt-in** `source-findings` mode produced three usable answers, one partial answer, one unnecessary abstention and one appropriate abstention under the same source-bound AI rubric. The existing 4B draft/review policy remains the default. This is a narrower evidence-quotation format, not a validated medical synthesis system.

All new relevance and answer judgments are **assistant-generated AI annotations (`human_authored=false`)**. Separate review contexts and a source-bound second pass improve development diagnostics; they do not constitute independent human or clinical validation. Genuine historical human labels, the blank human annotation package and held-out questions are unchanged.

## Measured behavior

The original six frozen questions and source snapshots were reused. Dense top-10 article order and source identities were preserved. Evidence packing and answer style changed. These questions were used repeatedly for development tuning, so the results are not held-out generalization evidence.

| Case | Baseline | Scoped findings | Remaining issue or covered target |
| --- | --- | --- | --- |
| bm01, heart failure | Partial | Partial | Sources report hospitalization results but do not explicitly establish adult eligibility; each visible card preserves the uncertainty. Mean age is not eligibility. |
| bm02, diabetes comparison | Partial | Requires correction | The add-on/dose-change contrast is excluded, but 4B omits the suitable qualitative review and leaves the answer empty despite sufficient packed evidence. |
| bm03, diagnostic accuracy | Requires correction | Usable | Culture-based adult findings retained; consensus-reference results excluded. |
| bm04, prognosis | Requires correction | Usable | AD-specific progression results retained; dementia-unspecified source excluded. |
| bm05, biological mechanism | Requires correction | Usable | Mouse/experimental-model context remains visible with separate positive and null findings; no human extrapolation. |
| bm06, original insufficient pool | Requires correction | Appropriate abstention | Analgesic/decongestant findings are not offered as antibiotic-versus-placebo evidence. |

Primary totals: **3 usable / 1 partial / 1 not usable / 1 appropriate abstention**. All ten visible source findings have factual support in the frozen sources; seven pass every applicable exact-question axis. Three retain unknown adult applicability and explicit cautions. Unknown is not counted as supported.

A separate seventh diagnostic uses the previously frozen, repaired PubMed-query snapshot for bm06: the antibiotic-versus-placebo pain result at two to three days is usable. It is not substituted for the original negative case, not included in the six-case aggregate, and receives no transferred historical relevance labels.

Six native selector calls completed with valid output and no JSON repair or output-cap failures; the original bm06 was rejected by program scope checks before a selector call. Every runtime was closed and reaped. Five primary cases with model selection took **15.13–17.67 seconds** each; the program-abstention case took **2.18 seconds**. The separate supplement took **15.94 seconds**. The previous six-case replay took 33.87–38.79 seconds. These are frozen-source replay costs including query encoding, fresh model load and cleanup; they exclude fresh PubMed download and passage-vector construction and do not establish live-request latency.

## What changed

1. The PubMed query planner removes only a recorded, conservative terminal outcome-follow-up phrase. Treatment duration, age/onset clauses, negation, quoted text and explicit search syntax remain protected. The unchanged full question still drives Dense and answer selection. Only bm06 changes its query among the six development cases. This is candidate broadening, not a claim to understand arbitrary PICO questions.
2. The opt-in evidence policy packs up to three articles with results and a methods/eligibility companion under six fragments, two per article and 4000 evidence tokens. All article ranks, actual window scores and source provenance remain recorded.
3. Quote boundaries are computed on each **full original source section** before intersecting the token window. A window-cut edge is no longer admitted as a complete quote. Full-section hashes and sentence spans are checked against the dataset; rehashing fabricated metadata cannot bypass source verification. Existing archived context catalogs retain their exact payloads and hashes.
4. Qwen selects bounded source IDs rather than generating factual prose. A separate schema field selects study setup before up to two findings. Recognized animal experiments require a visible actual-model quotation; unavailable bounded setup prevents a direct finding. Every selected quote remains exact, at most 532 characters; each card has at most three quotes and 1600 characters. Overlong source units are omitted whole and logged, never shortened or rewritten.
5. Deterministic lexical exclusions inspect the full verified title/abstract for explicit add-on/dose-regimen, reference-standard, progression-target, follow-up and co-intervention mismatches. They only demote sources and record exact proof spans. They are not a positive entailment or scientific-validity test. Missing adult eligibility generates a visible caution.
6. Reports show the source title, study-context quotation and finding quotations together. They explicitly say scientific support/completeness were not independently verified; they do not claim a model factual-review call occurred.

`answer.status` deliberately remains `partial` for every nonempty source-findings output: the program does not certify completeness. The external AI review category `usable` is a separate assessment of the observed output against the question and frozen sources. `quote_identity_verified=true` refers only to exact quotations; `semantic_support_verified` remains false.

## Rejected alternatives and limits

Shorter free-prose prompts and better results/methods coverage did not fix semantic reliability: one trial invented a study name and another reversed the direction of a clinical-resolution result. Quote selection alone also admitted the wrong comparison, reference and endpoint. Those trials remain preserved and are not presented as successful candidates.

Native thinking at 384 tokens failed to reach JSON on all eight balanced fictional controls; tested 768-token variants also capped. A brief comparison-note stage retained only 2/4 correct positive controls and accepted 1/4 negatives. Question-blind NLI scored 7/8 but lost a positive control. None was promoted. The research expanded beyond the initially planned small ablation sequence; all variants and failures are retained and the final result is disclosed as iterative development tuning.

The final lexical exclusions still do not prove that a matching time expression refers to the requested outcome, that a mentioned target was assessed, or that an adult word establishes eligibility of the actual sample. Synonyms, negation, multiple analyses, unusual text and study-role detection remain limitations. The boundary splitter is the transparent punctuation/whitespace heuristic, not a biomedical sentence parser. No clinical safety or unseen-topic reliability claim follows from zero observed explicit wrong-target findings in this small development run.

The **selection omission in bm02 remains unresolved**. A separate prospective V3 trial removed program-excluded articles before model attention. It recovered the useful comparison, but also displayed a tirzepatide GI/CV finding as an answer to the semaglutide-versus-glargine HbA1c question. Its six-case categories remained 3 usable / 1 partial / 1 not usable / 1 appropriate abstention, with one explicitly wrong-comparison/outcome card. The predeclared safety gate failed, so V3 was rejected and the exact V2 selector restored. No per-case mixture, relabeling or silent retry was used. Both trials and reviews remain intact. This is direct evidence that 4B semantic relation labels cannot be trusted as a standalone verifier. Adult applicability in bm01 is unresolved in these source abstracts. API work was deferred for this repair; the subsequent [local API](local-api.md) preserves these quality limits. This experiment does not automatically authorize release.

## Run the experimental mode

On the prepared macOS/Apple Silicon environment:

```sh
.venv/bin/python scripts/ask_live.py \
  --query "How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?" \
  --answer-style source-findings \
  --limit 100 --output-dir results/requests/source-findings-example
```

The same `--answer-style source-findings` option works with `scripts/ask.py --generate` and verified local dataset/window/embedding artifacts. `--answer-style draft` retains the current generation/review path. There is no model/backend fallback. The measured source-findings profile is Qwen3.5-4B through the pinned four-bit MLX artifact; other runtimes/models are not quality-validated by this experiment. Historical saved contexts must not be silently relabeled as the new complete-source-sentence policy.

## Reproducibility and checks

The [compact machine-readable record](experiments/biomedical-quality-improvement.json) contains protocol, implementation, run and review hashes. Local raw artifacts are under `results/biomedical-dev-v1/quality-improvement-20261112-v1`; the final trial is `source-findings-scoped`. Original baseline data remain under `quality-20261112-v1`.

The final prospective protocol SHA256 is `5c0c2a23ce83e6056d2a9f3363a94da5983cc62a186f92eccdad0e2accefe108`; the source-bound rubric is unchanged from baseline protocol `495ed18b8a135d7c48556a0847766acb5ea258992db1e34f6103ae005571451e`. The exact selector, runner and deterministic policy were frozen before this final generation run. No model calls or judgments are silently replaced.

Main suite: 472 tests, **463 passed and nine optional native-grammar tests skipped**. In the isolated native environment, all 22 structured-decoding/source-findings CPU tests passed. Native preflight checked all seven context configurations; the six selector calls used schemas that compiled and accepted their format fixtures within 6500 input tokens, 4000 evidence tokens and native request/schema limits. The restored V2 mode also completed a real local CLI request in 15.49 seconds with one selector call, three exact source cards, a closed runtime and byte-identical visible quotations to the separately reviewed bm05 output. This smoke test is excluded from the quality aggregate. Functional and identity tests do not replace the separate source-bound answer assessment.

Raw source records, quotations, model outputs, vectors and weights remain local. Only code, regression tests and compact experiment records are tracked.
