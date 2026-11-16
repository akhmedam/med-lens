# Source-selection quality research

H6 increased confirmed useful selections from 3/20 to 6/20 on the same updated
development sources. A separate, newly frozen twenty-question live pass also
yielded 6 useful selections and 1 unknown, but exposed four confirmed
comparison/outcome substitutions. **Release remains unaccepted.** Passing time
and refusal limits does not waive semantic support.

All new scientific judgments are **AI-authored**, not manual human annotation
or independent human/clinical validation. Historical genuine manual-human labels
were not modified or transferred to new source versions. The
[compact evidence record](experiments/usefulness-research.json) retains variant
identities, all twenty query decisions, source text-version and claim hashes,
timings, checks and hashes of the local original artifacts.

## Acceptance criteria

On 2026-11-16 the owner removed only the minimum useful-selection threshold.
The objective is the best measured quality among justified tested variants,
not proof of a global optimum. Retain at most **50% unnecessary abstentions
among source-confirmed answerable queries** and **60 seconds per complete
request**. A zero denominator is not an automatic pass; unknowns remain unknown.
Source-support and critical scope-error requirements remain. Historical useful
gates in frozen reports are not the current acceptance rule.

## Changes and rejected alternatives

The reference-role query fix recovered four formerly empty development searches;
only one output became useful. Private transport diagnostics retain sanitized
error classes and attempt timing. Eight old requests succeeded on the first
attempt of a later probe, but no causal reliability gain or historical
transport cause was established.

The selector distinguishes recognized placebo/control, disease presence versus
activity, qualified diagnostic target and clinical progression/regression
contracts. It retains recognized same-endpoint primary-null findings, permits
results under nonstandard content headings, and uses distinct finding IDs with
strict materialization. Adult-eligibility unknowns are excluded when adults are
required. Task guidance distinguishes effects, diagnosis, association and
mechanism, including applicable null findings. These checks can veto recognized
substitutions; they do not certify entailment.

Qwen receives the first Dense-ranked article with at most two fragments. Default
acquisition remains 100 candidates and ranked display remains top-10. One
bounded 4B/MLX call selects source sentences; model weights and quote caps were
not changed. Search/display and generation-context limits are distinct.

Informative top-10 packing regressed from 3 to 2 useful selections and from zero
to six confirmed critical errors and was reverted. History recognition added a
refusal without useful gain and was reverted. Early adult filtering added no
useful count over H4c and was not selected. Empty primary-null metadata perturbed
unrelated prompts; the retained correction omits it without an obligation.
Second-call rescue was not implemented because failure/checkpoint/deadline
integration required a wider refactor; no quality gain is claimed. Quote-cap
expansion was ruled out by a coverage audit, not measured as a native gain.

## Development comparison

| Variant | Useful + unknown /20 | Unnecessary refusals | Critical + unknown | Fixed-source median / max, s |
| --- | --- | --- | --- | --- |
| H0 | 2 + 6 | 5/12 | 1 + 6 | 20.56 / 29.67 |
| B1 | 3 + 6 | 7/15 | 2 + 6 | 23.51 / 33.89 |
| H2d | 3 + 6 | 7/15 | 1 + 6 | 27.28 / 39.13 |
| H2e | 3 + 6 | 7/15 | 0 + 6 | 29.84 / 55.26 |
| H2a | 3 + 6 | 8/15 | 0 + 5 | 26.33 / 50.30 |
| H2b | 3 + 6 | 7/15 | 0 + 6 | 27.11 / 58.06 |
| H2c | 2 + 6 | 1/15 | 6 + 7 | 32.62 / 84.15 |
| H2g | 3 + 6 | 7/15 | 0 + 6 | 26.03 / 38.10 |
| H3a | 2 + 8 | 5/15 | 2 + 8 | 30.64 / 42.97 |
| H3a2 | 3 + 6 | 8/15 | 0 + 5 | 29.58 / 41.58 |
| H3b | 4 + 2 | 9/15 | 0 + 1 | inherited; no new timing |
| H4a | 4 + 3 | 6/15 | 2 + 2 | 21.69 / 24.89 |
| H4b | 6 + 2 | 5/15 | 2 + 2 | 12.97 / 14.08 |
| H4c | 6 + 1 | 7/15 | 1 + 1 | 19.36 / 23.81 |
| H5 | 6 + 1 | 7/15 | 1 + 1 | 13.82 / 20.89 |
| H6 | 6 + 2 | 7/15 | 0 + 2 | 12.64 / 13.53 |

H0 has sixteen replay processes and four empty source pools. B1 replaces those
four pools: H0/B1 are not a pure selector comparison. B1-H6 share updated
source/case identities. H3b rematerializes inherited responses under exact model
inputs, with no new calls or timing evidence. Other rows are serial fixed-source
replays. Times exclude fresh acquisition and passage embedding and are **not
live request latency**. Unknown-label bounds are not statistical confidence
intervals. These questions are consumed development.

## Separate source-first live diagnostic

Candidate `00500b2ecc81a0210ca6d5feb2a61954a94addc4690f4fb0dd52452c8a228bbe`
was frozen before new questions were semantically opened. Each had one live
attempt. The source-only reference was frozen before output review: 16 sufficient,
3 insufficient and 1 unknown pool. This ordering does not establish independent
human blindness. Code was not tuned on these outputs. This is a new sample,
not a paired development delta.

| Check | Observed result |
| --- | --- |
| Useful selections | 6 yes, 1 unknown /20; descriptive bounds 30-35% |
| Unnecessary abstentions | 5/16 = 31.25%; bounds 31.25-35.29%; pass |
| Full CLI time | 20/20 within 60 s; median 21.00 s, maximum 41.44 s |
| Operations, source integrity and native cleanup | 20/20 pass |
| Critical scope | 4 confirmed, 2 unknown; fail |
| Displayed support / remaining checks | fail; release not accepted |

Failures are v2c03 (daily placebo instead of alternate-day versus daily iron),
v2c05 (adverse events instead of bleeding mortality), v2c14 (knee replacement
instead of incident osteoarthritis), and v2c17 (autophagosomes instead of
alpha-synuclein accumulation). v2c10 has an unestablished reference and lost
visualized-appendix selection; v2c19 has unresolved antecedent/endpoint context.
Both remain scope-unknown, not confirmed mismatches.

Read-only diagnosis of v2c05 found the mortality sentence, 379 characters and
below the 532-character cap, never entered the model payload. The FINDINGS window
retained a partial prior sentence and safety result; the only eligible finding
was E1.Q1. Zero quote-cap omissions did not establish full abstract coverage.
Window selection lost the result, and the model also misclassified the remainder.

## Verification and local artifacts

The main suite had 748 passes and 16 skips out of 764. Eleven native grammar
checks and four process-observation checks then passed in prepared environments:
**763 distinct passes, one optional Node.js UI probe skipped**. The wheel's
75 package files matched tested sources. Fresh installation, CLI/API help,
isolated imports and three HTML resources passed. Supplemental checks generated
no model responses; engineering success does not waive scientific blockers.

Frozen sources, raw outputs, original AI reviews, the verified research archive,
code snapshot/diff and Git bundle remain local under
`results/usefulness-ablation-20261116-v1/`, excluded from Git. The compact JSON
records their hashes; original reports/manifests are unchanged. This summary is
not a self-contained model/corpus bundle. New cases are now observed and cannot
serve as untouched confirmation after future behavior changes.
