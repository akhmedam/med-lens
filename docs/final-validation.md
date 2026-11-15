# Unified MVP and final validation

The frozen application passed all **12 operational cases**, but its semantic
quality is **not accepted for release**. Independent references are missing and
AI source-bound diagnostics found concrete question/source mismatches. Exact
quotations and successful execution did not make these findings correct answers.

## One supported application

[ADR 0007](adr/0007-unified-mvp-policy.md) and the [MVP guide](mvp.md) select
fresh bounded PubMed → BGE/MPS Dense → Qwen3.5-4B/MLX source-findings only.
Default candidates/top-k are 100/10. The public CLI has no answer-style/model/
backend selector; the API accepts only `source-findings`. Historical jobs retain
their original metadata. Draft, 9B, alternate rankers/backends and full-text
experiments are explicitly research tools. The BGE-only MVP setup and the
36-package main lock no longer require 9B, MedCPT or Transformers-Metal assets.
The separate native MLX lock is intentionally retained.

## Frozen confirmation

The [prospective protocol](final-validation-protocol.md) and
[machine-readable record](experiments/mvp-final-validation.json) bind the source,
prompt, package and asset identities. Freeze ID: `b2feceb605bad632427d40189381f872773b1601bfd3977b941c9fc15c1c73e7`.
Question texts were opened only after the freeze; prior inspection of candidate
PubMed expressions is disclosed. The original question/plan bytes are unchanged.
This separate cap100 operational pass does not replace the historical cap30/
four-ranker comparison. Every question was attempted once, in the original order,
without reruns, output selection, prompt edits or model tuning. Frozen source
hashes still match after the run. The twelve questions cover depression/anxiety;
they cannot establish broad biomedical, disease-disjoint or paper-disjoint quality.

| Case | Fresh articles | Answer status | Pipeline seconds | Citation occurrences | Technical checks |
| --- | ---: | --- | ---: | ---: | --- |
| h01 | 100 | partial | 33.43 | 8 | pass |
| h02 | 100 | partial | 34.18 | 9 | pass |
| h03 | 100 | insufficient_evidence | 33.56 | 0 | pass |
| h04 | 9 | partial | 18.62 | 5 | pass |
| h05 | 100 | partial | 30.68 | 5 | pass |
| h06 | 49 | partial | 26.06 | 4 | pass |
| h07 | 27 | partial | 23.30 | 5 | pass |
| h08 | 28 | partial | 24.02 | 6 | pass |
| h09 | 100 | insufficient_evidence | 45.29 | 0 | pass |
| h10 | 44 | partial | 27.72 | 5 | pass |
| h11 | 38 | partial | 23.29 | 6 | pass |
| h12 | 43 | insufficient_evidence | 25.78 | 0 | pass |

All 738 acquired candidate records were
retained. The 68 selected passages and
53 visible citation occurrences
passed source-binding checks. Public/private projection, exact text versions,
source offsets and same-article bindings passed. All twelve native workers were
observed and reaped. Nine partial selections and three abstentions are execution
outcomes, not nine good answers or three verified appropriate refusals.

## Checks and performance

The full main suite passed **552 tests, with 9 optional native skips (561 total)**.
The isolated MLX environment separately passed all **11 structured-decoding** and
**11 source-findings** tests; these overlap the main suite and are not 22 new
unique checks. `pip check`, the BGE asset checksums and real CLI/API default parity
passed. API draft input returns 422 before job admission. Existing deterministic
tests cover malformed/control input, empty search/evidence, upstream/model errors,
corrupted cache, no stale fallback, busy admission, cleanup/storage failures,
partial results and safe exports. They are synthetic failure checks, not fresh
clinical examples. Language/topic classification remains absent by explicit
policy; accepted out-of-scope text is not counted as a successful scope refusal.

The held-out pipeline times were **18.62–45.29 s**,
median **26.89 s**. On one preselected development question,
the dedicated empty-passage-cache request took **23.14 s**
(31 new inputs); repeats took **17.86** and
**18.11 s** (31 cache hits each). This is not a cold OS/model
startup comparison, a latency SLA or a causal estimate of cache speedup.

Native Metal peak reached **3.71 GiB** and native process RSS
peak **2.79 GiB**. These overlap and must not be added.
Current parent RSS after jobs increased from **771.1 to
995.1 MiB** across the twelve different requests. Native cleanup is
confirmed; parent allocator/cache retention versus a long-run leak is unresolved.
The sampled parent also runs source-index audits and retains measurement records;
the growth cannot be attributed to the application alone. This run establishes
neither whole-machine/swap peaks nor long-session stability.

## Semantic diagnostic and missing reference

The twelve frozen outputs received a separate **AI-origin** source-bound diagnostic
(`human_authored=false`). It creates no human labels, qrels or quality score.
Examples requiring correction/review:

- **h01/C2, PMID 35952325:** reflective-functioning change is used for a depression-
  symptom question. The abstract says: “Change in RF was not correlated with
  change in depression.”
- **h04/C1, PMID 21898715:** relapse predictors and acute combination-versus-
  monotherapy are used for continuation-versus-discontinuation of antidepressants.
- **h05/C3, PMID 42029097:** antenatal prevention is used for treatment of existing
  postpartum depression, without that setting in the displayed card.

Other cards retain uncertain adult eligibility, combined-treatment attribution,
indirect results or incomplete setup. Abstention appropriateness is not inferred
from empty output. The AI observations do not certify the unflagged cards.
Complete local diagnostic records bind their source proofs and unchanged outputs.

The candidate pool has 727 distinct PMIDs; 7
also occur in the preserved pilot human-label PMID pools, and
0 in the frozen broader-dev candidate pool. These are
identifier overlaps, not transferable labels or verified original text identity.
No finalized independent relevance/claim/answer reference exists for these new
sources. nDCG/Recall/MRR and independent answer-quality metrics remain null.

## Remaining work and release decision

The supported pipeline and its options are now unambiguous. Windows/Linux/CUDA,
CPU-only Qwen, other models/rankers, PMC full text, a larger Medium/ANN service
and a web UI are outside this MVP; historical research records are preserved.

**Release quality conditions remain unsatisfied.** Semantic selection defects and
the missing independent reference remain blockers. Packaging/CI, clean main/MLX
installation and longer-session parent-RSS investigation are explicit remaining
engineering work. Those tasks cannot substitute for semantic validation. After
observing these test outputs, a behavior change needs a new untouched confirmation
set; the present results must remain intact. Raw source texts, weights, vectors,
prompts, outputs, diagnostics and test logs stay in ignored local artifacts.
