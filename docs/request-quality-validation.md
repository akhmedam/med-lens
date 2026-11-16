# Request reliability and actual-output validation

The repaired candidate passed the observed complete-request timing and process-cleanup checks. **It did not pass release acceptance** on the reserved twenty-question live series. The owner delegated the source and output reviews to the assistant: all new semantic judgments are **AI-origin**, with `human_authored=false`. This is neither independent human nor clinical validation.

## What changed

Each CLI/API request now owns a short-lived process group. One ingress clock reserves 55 seconds for work, three for termination/reaping and two for delivery. Unknown cleanup closes admission; timeout is failed execution with `not_completed`, rather than a completed evidence abstention. The API parent retains control state and disk artifacts instead of model/tensor runtimes. MLX distinguishes a closed interface from confirmed process release, including repeated cleanup after an uncertain exit.

The one-call 4B selector uses explicit applicability decisions and exact source IDs. Study setup has priority over procedural boilerplate. Recognized checks demote population/comparison substitutions, endpoint-change correlations presented as treatment effects, prevention presented as treatment, and objectives/procedures presented as results. These repairs were developed on consumed development material before the candidate freeze. They do not constitute a general semantic verifier.

## Observed engineering checks

| Check | Result |
| --- | --- |
| Main suite | 640 passed, 11 skipped; 651 total |
| Selected native-environment checks | 111 passed; overlap with the main suite |
| Dependency consistency / wheel | Passed; 75 packaged Python/HTML files matched source bytes |
| Stable development CLI/API stress | 29/29 operational successes; maximum 49.34 seconds |
| Same API process, twenty requests | Parent RSS 52.36→54.64 MiB; +2.28 MiB |
| Additional actual-300-article probe | 23.76 seconds; fresh 300-record snapshot/dataset, one Qwen runtime load, completed evidence abstention, cleanup confirmed |

The predeclared boundary matrix used `limit=300` but actually acquired 66 articles. The separately planned additional broad-query probe verified 300 actual articles; it is a post-hoc development observation, not a scientific-quality or universal worst-case result. All model experiments ran sequentially. The first driver attempt and its API startup-command error are retained separately from the complete stable re-run.

Current RSS samples recorded access-denied gaps in 26 stress rows. Parent idle points were observed for all twenty API requests; request/native-worker peaks are sampled observations, not upper bounds. No owned request descendants remained in the observed cleanup proofs. This finite series does not establish unlimited session stability, whole-machine memory bounds or a universal client-latency guarantee. See [implementation and limits](request-reliability.md).

## One actual live confirmation

Before execution, the application, harness, policy, models, tokenizers and environments were frozen as manifest `a870a964853396ee16b2163e06b5e3dbd357e5ff97985b539f9298bc447f054c`. The fixed plan was then opened once, in its original order, through real CLI subprocesses with fresh PubMed acquisition and the supported default limits. All twenty attempts remain in the denominator; failures were not retried as replacement observations. Complete time includes CLI startup, stdout receipt, process exit and cleanup. Post-request source inspection is outside that clock.

| Acceptance item | Result |
| --- | --- |
| Useful selections, minimum 10/20 | **1 yes, 4 unknown, 15 no**; possible range 5–25%; fail |
| Unnecessary abstentions, maximum 50% of answerable cases | **3/7 (42.9%)** of confirmed answerable queries; possible rate 17.6–42.9% with unknown source answerability; pass |
| Full-request time, maximum 60 seconds each | **20/20**, maximum **44.67 seconds**; pass |
| Native/process cleanup | **20/20 confirmed**; pass |
| Successful complete acquisitions/executions | **12/20**; eight transport failures; fail |
| Confirmed critical scope mismatches | Zero confirmed, four unknown; absence not established |
| Displayed-card support/qualifiers | Fail: one important primary-null omission; another card retains unresolved trajectory context |
| Fresh source integrity/coverage | No proven altered source; eight unavailable actual pools prevent complete establishment |

Twelve requests succeeded: six nonempty selections containing fourteen cards and six scientific abstentions. Queries hc09–hc16 failed at ESearch after three transport attempts. No specific HTTP status or underlying transport cause was retained; calling this a proven PubMed server outage would exceed the evidence. They remain execution failures, not appropriate evidence refusals. A later successful development probe does not repair those observations.

Source reconciliation preserved 127 exact question/PMID/text-version matches and separately reviewed three new articles. Eleven pools changed, including the eight unavailable ones. Changed available pools were read using full primary and supplementary abstracts before source answerability was finalized. Only then were actual model cards exposed for separate review. The finalized actual-pool reference has seven yes, three no and ten unknown answerability decisions. The operator had previously helped prepare the original AI reference; the separate exposure declaration prevents presenting this as an independent blinded human study.

The sole fully usable selection is hc17 (high-risk pregnancy/aspirin), including its null trial. hc03, hc06 and hc08 retain unresolved adult eligibility in one or more cited abstracts. hc19 retains an opaque trajectory-specific numerical card whose groups/eligibility cannot be fully reconstructed from the shown extract. These are unknowns, not confirmed wrong populations. hc05 accurately quotes an explicitly exploratory positive result but omits that study's nonsignificant primary analysis (`P=0.067`); its selected sentences are exact, yet the selection loses material counterevidence. Three source-answerable questions (hc01, hc04, hc20) were refused. The three empty diagnostic pools do not establish that the biomedical questions lack relevant PubMed evidence.

## Consequences and reproducibility

Timing, cleanup, source identity and scientific usefulness are separate gates. The overall AI gate is **fail**. The candidate remains experimental; cleanup/CI work cannot by itself approve release. Further repair must address source-acquisition reliability and empty candidate pools, selection omissions and retention of material null/qualifier context. Actual-output feedback from this round is now consumed; if it guides tuning, use another untouched confirmation set to assess the next candidate.

The [compact machine-readable record](experiments/request-quality-validation.json) contains all twenty timings/decisions and exact manifest/report identities. Local raw snapshots, private/public jobs, stdout, process proofs, original AI exports and preserved failed attempts are under `results/request-quality-fixes-20261116/`; they remain separate from historical genuine human labels and the original blank human form. Original protocols and historical measurements were not rewritten.

The repeatable harness is `scripts/validate_confirmation.py`: `freeze`, one-shot `run`, exact-source `reconcile`, source-only `finalize-sources`, then `review-template` and `finalize`. Source relevance alone never supplies output-usefulness labels. An output review must retain the frozen source answerability, explicit unknowns, per-card facets and all twenty attempts.
