# Day 21: semantic support diagnostics

The fixed development experiment did **not** justify promoting the seven-axis
review candidate. The production pipeline preserves `claim-support-v4`;
`claim-support-v5` is explicitly experimental. Same-model review remains fallible,
and neither protocol is independent clinical validation.

## Source inspection

The [inspection record](experiments/day21-source-inspection.json) covers all 22
visible claims from the recovery pilot. It binds the exact raw-report hash and
separates explicit textual/answer-scope mismatches, missing source scope, and
claims for which this inspection identified no specific issue. These are AI
source comparisons, not clinical gold labels or a measured clinical accuracy.

Examples include dysfunctional attitudes relabeled as depressive symptoms,
antidepressant-class evidence narrowed to SSRIs, recommendations/background used
as efficacy results, and within-group improvement presented as a treatment effect.
Several excerpts also omit explicit adult eligibility. Absence of that scope does
not prove the underlying clinical statement false; it limits what the displayed
sources can support. Exact quotation checks continue to pass despite these errors.

## Protocols

The preserved v4 reviewer sees cited excerpts and checks population, comparison
and outcome. The experimental v5 reviewer checks seven mandatory axes:

- population/age/diagnosis;
- exact intervention;
- comparator;
- measured outcome;
- age, severity, subgroup, time and other qualifications;
- numerical, causal, direction and null-result fidelity;
- consistency with all selected excerpts, including conflicting findings.

Only comparison may use `not_requested`; all other axes must match. An explicit
supported verdict is also required. Unknown fields/enums, omitted claims,
nonfinite or duplicate-key JSON and output/input overflow remain errors. Claims
are never rewritten or repaired. V5 sees the complete selected quote catalog;
uncited text may expose restrictions/conflicts but cannot replace a missing
supporting citation. Whole-library evidence is outside this scope.

The generator stays at draft v6. Qwen revision, Metal 4-bit/group-64, FP16 runtime,
768/512 draft/review output caps and 4000/6500 evidence/input limits are unchanged.
One model instance is reused sequentially; no CPU inference fallback or allocator
change is introduced. Real-tokenizer packing with 4000 evidence tokens and three
synthetic claims used 4774 draft / 4945 review input tokens. Fake model outputs
were used for this packing check; it is not inference or semantic validation at
that budget.

## Controlled development contrasts

The tracked [fixture](../benchmarks/semantic-dev-v1/cases.json) contains seven
explicitly fictional groups with one positive and two negative claims each.
It covers outcome substitution, age/severity, class/diagnosis substitution,
indirect comparators, pre/post and follow-up inference, null findings,
contradictions and citation laundering. No source is a real clinical study.
Labels and focus tags are omitted from model prompts. Fixtures and prompt hashes
were fixed before inference; no post-result prompt tuning was performed.

The [paired result](experiments/day21-semantic-challenges.json) records:

| Protocol | Supported controls retained | Unsupported claims accepted | Invalid review groups |
| --- | ---: | ---: | ---: |
| v4 | 7 / 7 | 1 / 14 | 0 / 7 |
| v5 candidate | 6 / 7 | 3 / 14 | 0 / 7 |

V5 newly accepted equivalence from nonsignificance and a GAD-diagnosis claim from
a source explicitly lacking that diagnosis assessment. Both versions accepted a
claim whose supporting outcome appeared only in an uncited source. V5 also
withheld a faithful statement of conflicting study results despite marking all
axes as matching and using a supported reason code. Its `not_supported` verdict
was retained; code did not repair this inconsistency into an accepted claim.

These are synthetic development counts, not general biomedical accuracy.
The candidate fails the practical promotion requirement: do not introduce newly
accepted unsupported controls or lose supported controls merely to add axes.

## Frozen real drafts

The [review-only comparison](experiments/day21-semantic-comparison.json) uses the
same questions, context/profile identities, model/packages, draft prompt hashes,
raw draft texts and token budgets as Day 20. Cached drafts are reused; only review
calls are fresh. The previously overlong p06 draft remains a format failure and
supplies no parseable claims to this semantic comparison. All ten pilot cases and
the two controls remain in the case table. Draft-generation quality and retrieval
quality are not remeasured.

Across 26 parseable draft claims in nine review pairs, v5 kept 18 of the 22
baseline-visible claims, withheld four and accepted no previously hidden claims.
Both controls still abstained; p06 retained its original output-cap failure.
The candidate produced three answered and six partial pilot drafts. Offline
replay reproduced all twelve cases using twenty saved calls, and all 25 remaining
visible citations matched source spans; all 284 catalog entries resolved.

Of five explicit textual/answer-scope issues identified by AI source inspection,
the candidate withheld the antidepressant-to-SSRI extrapolation and the background
assertion used as an efficacy result. Three remained visible: the DA/outcome
substitution, the recommendation used as an outcome, and the pre/post intervention
conflation. Eleven of thirteen source-scope uncertainties remained visible.
All four claims for which inspection found no specific issue were retained;
this is not an independently measured correctness rate.

The final default-policy audit reproduced all original twelve v4 outcomes,
claims, reviews and rendered answers from twenty cached calls, with 29 citations
checked against sources. The [check record](experiments/day21-semantic-checks.json)
reports 260 passing unit tests, dependency checks, protocol-cache separation,
all mandatory axes, input overflow without source truncation and unchanged
production defaults.

The comparison bundles changed instructions, axes and review evidence visibility;
it cannot attribute an outcome to any one mechanism. Inspection-derived error
counts are descriptive AI findings, not expert-adjudicated false-positive rates.
Reused draft component times are historical; fresh review times are reported
separately and must not be presented as a fresh end-to-end latency benchmark.

## Reproduce

With the pinned assets and rebuilt collections available, use fresh result paths:

```sh
.venv/bin/python scripts/evaluate_semantics.py \
  --output results/semantics/new-challenges.json \
  --summary results/semantics/new-challenges-summary.json
.venv/bin/python scripts/generate_answers.py \
  results/requests/recovery-20261107-pilot \
  --collections configs/collections/recovery-20261106.json --include-controls \
  --review-protocol claim-support-v5 \
  --reuse-outputs results/generation/recovery-20261107-v6.json \
  --output results/generation/new-review-v5.json \
  --summary results/generation/new-review-v5-summary.json
```

For the recorded experiment, the comparison command also pins its exact candidate
implementation commit because the final default-preservation change came later:

```sh
.venv/bin/python scripts/compare_semantic_reviews.py \
  results/generation/recovery-20261107-v6.json \
  results/generation/day21-20261107-review-v5.json \
  --requests results/requests/recovery-20261107-pilot \
  --collections configs/collections/recovery-20261106.json \
  --candidate-ref 7f6dfbe59e112c2a9befa7eeb4aeaad51b121a37 \
  --output results/semantics/new-comparison.json
```

Omit `--candidate-ref` only when the checkout exactly matches the candidate's
recorded code hashes. The comparator rejects changed drafts, cases, controls,
models, packages, resource identities and budgets, and replays current explicit
v5 prompts/decisions from saved calls without inference. Public reports contain
no source article bodies or raw model text; local raw reports remain ignored.

## Remaining work

Both reviewers still make material support errors. More fields and valid JSON do
not establish entailment. The synthetic groups expose only selected failure modes;
coverage of dose, numerics, ambiguous language and conflict completeness is limited.
Review cannot recover sources absent from retrieval or prove that every contrary
finding in the selected context was represented in an answer.

Reserve work should test focused per-claim evidence extraction and separate the
instruction/axis/context changes on fixed development cases, then obtain
independent human assessment. Do not tune on `benchmarks/heldout-v1`. The original
Day 13 raw data remain unavailable, so this experiment uses the rebuilt pilot.
Day 21 delivers checks and an explicit error record; release semantic conditions
remain unmet. Corpus work on Day 22 can proceed without treating these findings
as clinical readiness.
