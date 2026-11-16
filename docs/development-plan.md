# Development plan

The supported [MVP application](mvp.md) is bounded fresh PubMed acquisition →
BGE/MPS Dense ranking → exact evidence → Qwen3.5-4B/MLX source findings.
[ADR 0007](adr/0007-unified-mvp-policy.md) fixes one source-quotation format for
CLI/API. Findings remain partial and independently unverified. The current
selector is `source-findings-task-scoped-v5-h4a-task-aware-specific-comparison`, with one bounded
applicability decision and exact sentence selection; earlier development
policies remain historical. Draft/support-review, 9B and other rankers remain
research tools.
Earlier measurements retain their original scope and do not approve autonomous
answers or establish cross-topic clinical quality.

The [broader biomedical stage](newbroader-biomedical.md) prepared six frozen
English cross-topic development cases, 246 candidate records and a verified
84-card blank human relevance package. See the
[preparation/integrity record](experiments/biomedical-scope-preparation.json).
The user subsequently requested assistant annotation of all questions. A separate
local assistant package now contains all 84 judgments; it is explicitly
`human_authored=false`, not an independent human reference. The original blank
human package and historical genuine human labels remain unchanged.

On 2026-11-11 the user allocated one working day to broader-topic development
quality assessment, completed on 2026-11-12 in the
[exploratory assessment](biomedical-quality.md): two partial answers, four requiring
correction, and material semantic-support errors still open. Retrieval diagnostics
use the completed assistant labels with their provenance disclosed. Generated answers require a separate claim-level
review against the exact source abstracts; article relevance alone cannot score
answer correctness. This exploratory assessment does not satisfy the independent
human-reference gates recorded in
[ADR 0006](adr/0006-biomedical-scope-and-human-reference.md).
The existing held-out set remained unopened during that preparation; the separately frozen final application pass occurred on 2026-11-15.

On 2026-11-12 the user deferred merging and prioritized quality repair before the
API. The [scoped-findings experiment](biomedical-quality-improvement.md) adds an
explicit opt-in source-quotation mode with three usable development answers;
the then-current draft/review defaults remained unchanged. Selection omissions and unreliable
semantic relation judgments remain open. API work was paused behind this quality
repair until that change was reviewed. Its unresolved semantic issues remain
release constraints.

The quality-repair PR was subsequently merged. On 2026-11-13 the next delivery
stage implemented the [local asynchronous API](local-api.md), with one active
job, persisted progress/results, explicit errors and verified runtime cleanup.
The API exposes the existing experimental workflow; remaining semantic-quality
issues and independent-human gates remain release constraints.

On 2026-11-14 the [interactive CLI](interactive-cli.md) added a plain-text question
loop, stage progress, cited results, evidence display and public HTML/JSON export.
It reuses the same local service and model policy without an HTTP server or a new
CLI dependency. Real input/export and graceful Ctrl+C checks passed; final
held-out validation remained a separate subsequent stage.

On 2026-11-15 the owner requested final checks and unambiguous MVP options.
The application policy was consolidated before opening the test questions:
source-findings only, PubMed default100/top10, BGE/MPS and Qwen4B/MLX. Historic
scripts and platform alternatives are explicitly research/post-MVP. The
[prospective operational protocol](final-validation-protocol.md) preserves the
original held-out files and declares a separate twelve-case application pass;
its results cannot fill the missing independent retrieval or answer references.
The [final validation report](final-validation.md) records 12/12 operational
passes, 53 source-bound citation occurrences, nine partial selections and three
abstentions. A separate AI diagnostic identifies clear scope mismatches; it is
not a human quality reference. Independent quality remains unestablished and
release conditions are unsatisfied. For that frozen pass, source/model/prompt
policy remained unchanged. The subsequent repairs below use already observed
material as development and require a new confirmation freeze. Parent current
RSS grew during the historical series despite native cleanup; its measurements
do not describe the later isolated-request lifetime.

The [2026-11-16 quality-policy amendment](usefulness-research.md#acceptance-criteria)
sets the current quality objective to **the best measured quality among the
variants tested through sequential evaluation of the justified hypotheses**.
Report useful-selection counts and denominators, explicit unknowns, null effects,
regressions and failures. There is **no minimum useful-selection percentage or
count for MVP readiness**. The separate limits remain at most 50% unnecessary
abstentions among reference-confirmed answerable queries and a maximum of sixty
seconds for each complete request; source-quality facets and critical scope-error
blockers are unchanged.

The [human-confirmation record](human-confirmation.md) preserves the original
protocol and identifies its earlier useful floor as historical. The original
[frozen plan](../benchmarks/human-confirmation-v1/plan.json), source forms, results
and reviews remain unchanged. Preparation generated no Qwen answers or human
grades; subsequent AI annotations keep their separate provenance. Actual-output
quality review still requires source reconciliation and a frozen candidate.
The isolated-request implementation enforces 55 seconds for work, three for
cleanup and two reserved for delivery; previous timing observations are not
retroactively declared a pass of the complete-request limit.

The owner subsequently delegated source annotation to the assistant because
manual annotation was no longer practical. A separate local AI-origin assessment
now covers all 208 article/query pairs and all twenty answerability decisions:
12 sufficient, six insufficient and two uncertain source pools. Two blind readings
of the primary abstracts, source-bound adjudication and a separate supplementary-
abstract audit are complete. The original blank human package and frozen plan
remain unchanged. These labels do not assess Qwen's actual selections or establish
independent human validation; the later output review must retain its true origin.

The owner expanded the remaining schedule on 2026-11-15: day 30 addresses
functional defects and acceptance checks, day 31 covers cleanup, CI and
documentation, and day 32 covers clean installation and conditional release.
The historical numbering therefore continues through development days 26–32.
The assessment, API, CLI and final operational checks through day 29 are complete;
semantic release gates remain unresolved independently of that schedule.
These are working-session targets, not a readiness guarantee. Material unresolved
support, deadline or resource-cleanup defects can delay release. The earlier Air
recovery, live-search, PMC and ranking records remain useful evidence within
their documented limits.

Current repairs are documented in [request reliability](request-reliability.md).
The CLI/API use `medlens-mvp-source-findings-v2` and a fresh owned process group
per accepted request. A deadline is a failed execution with `not_completed`
answer status, not an insufficient-evidence selection. Source repairs distinguish
actual outcomes from objectives/procedures, retain study setup, and veto recognized
population/comparison mismatches. Consumed development cases still expose false
abstentions, uncertain adult eligibility and internally inconsistent sources.
The [actual twenty-question AI validation](request-quality-validation.md) on
2026-11-16 passed all full-request timing and cleanup checks but failed release
acceptance: one confirmed useful selection, four uncertain selections, eight
PubMed transport failures, and a displayed primary-null omission. The unnecessary-
abstention bound passed at 3/7 confirmed answerable queries. Functional repairs
and this first confirmation are recorded; source-acquisition reliability and
semantic-quality blockers remain before release. Cleanup/CI cannot waive them.

The primary user interface is a plain-text interactive command-line application.
HTML reports remain an optional export. The API is a separate programmatic
interface; a web UI is outside this release plan.

The later [local sequential quality study](usefulness-research.md)
is complete. H6 preserved six useful development selections while removing two
known endpoint substitutions. On a newly frozen once-only live sample it produced
6/20 useful selections (1 unknown), 5/16 unnecessary abstentions (bounds
31.25–35.29%) and four confirmed comparison/outcome substitutions. All twenty
complete requests finished within sixty seconds (maximum 41.44 s); source
integrity, operations and cleanup passed. The useful minimum was not applied.
Semantic support remains a release blocker; cleanup/CI and the passed time/refusal
checks do not waive it. These new cases are now observed and cannot be reused
as untouched confirmation after further tuning. Changes remain local.

## Historical development sessions

| Day | Work | Deliverable |
| --- | --- | --- |
| 15 | Pin MedCPT model/tokenizer revisions; test CPU/MPS, batching and the 512-token pair limit. Fix the held-out questions before tuning. | A measured question/text scorer with explicit input limits and no silent source truncation. |
| 16 | Implement BGE dense top-10 articles followed by MedCPT, and direct MedCPT over all candidate articles. Score every window of each eligible article and aggregate by maximum. | Two reproducible modes on identical snapshots and source windows. |
| 17 | Compare PubMed Best Match, Dense, Dense → MedCPT and direct MedCPT. Judge newly retrieved results absent from the current pool. | Comparable nDCG@10 and Recall@10, with error analysis and incomplete judgments identified. |
| 18 | Vary candidate-pool size; measure latency, pair count and memory. Separate Dense with cached embeddings from building embeddings. | A quality, latency and memory comparison on the M3 Pro. |
| 19 | Compare measured options, record the user’s choice and prepare transfer to the Air M3 (24 GB). | [Dense → Qwen9B accepted](adr/0003-primary-pipeline.md), existing request integration checked, and [portable handoff](handoff.md) verified; [subsequent Air recovery checks](experiments/air-recovery.json) completed. |
| 20 | Improve generation format reliability: bounded claims/citations and structured output; measure refusals against Day 13. | [New-corpus format baseline measured](day20-format-reliability.md), strict gates retained; output-cap work remains. |
| 21 | Evaluate population/age, severity, intervention, comparator, outcome, null findings and contradictions. | [Paired review diagnostics and remaining-error record](day21-semantic-reliability.md); v5 remains experimental after regressions. |
| 22 | Pool local collections, handle duplicate records and text versions, verify cache updates and start the Small → Medium transition. | [Shared collection and immutable cache](shared-corpus.md): 277 unique articles, 850 windows; preparation, cache lifecycle and resident search measured. Larger Medium and relevance validation remain open. |
| 23 | Test a small explicitly licensed PMC subset, preserve JATS section/source provenance and date filters, and compare paired abstract/full-text costs and rankings. | [Feasibility result](pmc-fulltext.md): full text remains opt-in; no judged quality gain is established. |
| 24 | Implement live PubMed retrieval within the existing depression/anxiety scope, using bounded candidate pools and cached unchanged embeddings, then run Dense, evidence selection and Qwen generation/review in one request. Implemented on 2026-11-10. | [Live pipeline and measured costs](live-search.md): twelve fresh cold/warm requests at 30/100/300 candidates; exact source checks and 288 passing tests. Format failures and abstentions remain visible; answer quality is not scored. |
| 25 | Prepare broader English biomedical question handling and a blank source-bound manual annotation package; subsequently complete a separate assistant annotation at the user’s request. | [Frozen six-case plan and verified 84-card blank package](newbroader-biomedical.md) completed; all 84 assistant judgments are complete. Independent human annotation remains pending; quality assessment was performed separately afterward. |

## Remaining MVP schedule

These are development days 26–32, continuing the 25 completed working days.
They are working sessions, not calendar deadlines.

| Working day | Work | Deliverable |
| --- | --- | --- |
| 26 | Assess quality on the six frozen broader-topic development questions within one day. Completed 2026-11-12. | [Exploratory AI assessment](biomedical-quality.md): Dense nDCG@10 0.7888 versus PubMed 0.5376 (5 defined cases); two partial answers, four requiring correction. Fixed-source replay 33.87–38.79 s; separate query-override live request 44.75 s. Human-reference gates remain unsatisfied; semantic-support defects are explicit release blockers. |
| 27 | Add a minimal programmatic API with request parameters, structured results, visible errors and model lifecycle management. Completed 2026-11-13. | [Local asynchronous API](local-api.md): one live job, progress/results, explicit busy/error/abstention states, graceful cleanup; fresh PubMed/Qwen4B HTTP smoke verified. |
| 28 | Build the interactive CLI: question input, progress, ranked articles, evidence, answers/citations, errors and optional report export. Completed 2026-11-14. | [Plain-text interactive CLI](interactive-cli.md): one command, visible statuses/citations, public HTML/JSON export and verified graceful shutdown on macOS/Apple Silicon. |
| 29 | Freeze one MVP policy and run final source/contract/runtime checks. Completed 2026-11-15 without held-out tuning. | [Final validation](final-validation.md): 12/12 operational passes; semantic mismatches and absent independent references keep release quality unaccepted. Memory retention limits are explicit. |
| 30 | Repair the known population/comparison/outcome selection defects on old development or already observed test material. Implement one sixty-second complete-request deadline with cancellation and actual cleanup. Investigate and fix or bound longer-session parent-RSS retention. Freeze the repaired candidate, run all twenty confirmation questions through the complete live path, reconcile changed sources, and separately review actual outputs and abstentions. | Recorded usefulness and unnecessary-abstention numerators/denominators, per-request wall time, source/scope checks and resource evidence. Quality objective: the best measured quality among sequentially tested justified hypotheses, with useful counts and unknowns reported and no useful percentage/count readiness floor. Separate limits: at most 50% unnecessary abstentions among reference-confirmed answerable queries and every accepted full request at most 60 seconds. Preserve failures and uncertainty; disclose AI assessment provenance. |
| 31 | Clean and consolidate code, separate supported MVP dependencies from research tools, configure CI for offline regression/build/install checks, and update setup, usage, limitations and annotation-provenance documentation. Preserve historical experiment records and the evaluated application behavior. | A maintainable release candidate with passing CI, verified packaging and current documentation; regression and behavior-parity checks after cleanup. Any behavior-changing edit reopens acceptance validation. |
| 32 | Follow setup from a clean checkout on Apple Silicon, verify pinned BGE/MLX model preparation, exercise CLI and API end to end, and prepare versioned release artifacts and release notes. | A reproducible portfolio v1.0 release only after quality, full-request latency, source-integrity, resource-cleanup and installation checks pass; otherwise retain a release candidate with explicit blockers. |

Day 30 repairs precede candidate freeze. The live confirmation uses the actual
outputs and their complete elapsed time; source relevance alone and old download
timings cannot substitute for that evidence. Fresh PubMed sources require exact
question/PMID/text-version reconciliation and renewed answerability decisions
when the reviewed pool changes. Unknown decisions remain explicit, and all
twenty attempts stay in the usefulness denominator. Deadline stress checks also
cover the accepted candidate/top-k boundaries and cold/warm passage-cache states
on separate development inputs.

Day 31 cleanup must preserve the evaluated behavior and record the resulting
code/package identities. If a refactor or dependency change affects selection,
generation, timing or cleanup, its previous acceptance result is no longer
sufficient. If confirmation cases or their labels guide further tuning, record
that use as development and obtain another untouched confirmation round rather
than reusing the same cases as an independent test. The release date does not
waive unresolved acceptance conditions.

## MVP platform and post-MVP portability

The user narrowed the MVP platform scope on 2026-11-11. The supported MVP
platform is macOS on Apple Silicon, using the existing BGE retrieval pipeline
and Qwen3.5-4B through MLX. Clean-install instructions and end-to-end release
checks target this platform.

Cross-platform support is deferred until after MVP. Windows/Linux,
NVIDIA/CUDA, CPU-only Qwen execution and automatic selection between
platform-specific runtime backends are outside the MVP acceptance criteria.
They are not prerequisites for the API, interactive CLI or portfolio v1.0.
Future portability work must preserve source/citation checks, output contracts
and evaluation provenance, and verify each platform before claiming support.

## Completed: live PubMed search on 2026-11-10

Accepted by the user on 2026-11-09. The latest scheduling decision keeps this
session within the existing depression/anxiety topic scope. The implementation
and [runtime/integrity record](live-search.md) are complete. Broader biomedical
coverage and its new manual annotation round begin in the following session on
2026-11-11. The intended live request becomes:

```text
Question -> PubMed query -> fresh candidate search and abstract download
         -> parsing and source windows -> cached/new BGE vectors -> Dense top-10
         -> exact evidence -> Qwen draft and support review -> answer with citations
```

Completion criteria:

- One command accepts a new question and performs a live PubMed search; a manually
  selected local collection is no longer required for that mode. Existing local
  artifacts remain usable for reproducible offline experiments.
- Define and document question-to-PubMed-query construction, including any explicit
  query override. Record the question, submitted query, PubMed query translation,
  candidate count, retrieval timestamps and snapshot identity in the request report.
- Fetch current candidate records for each live request. Apply an explicit candidate
  cap rather than promising to retrieve every matching article. Compare 30, 100 and
  300 candidates on fixed development questions and select a practical default from
  the measurements; no latency target is claimed before that experiment.
- Reuse exact unchanged token inputs from the verified embedding cache. New or
  changed inputs are encoded with the pinned BGE policy. Keep model lifetimes
  bounded on the Air M3, 24 GiB, through sequential BGE and Qwen execution.
- Run the existing Dense, evidence, Qwen generation and default support-review
  stages, retaining exact source citations, validation gates and explicit
  insufficient-evidence outcomes. API errors and incomplete downloads are visible;
  cached data must not be silently presented as a successful fresh search.
- Measure the complete request with cold and warm embedding caches, separating
  search/download, parsing/windowing, model loading, passage/query encoding,
  ranking/evidence packing, generation and review. Report process and device
  memory with their scopes. Preserve snapshots and raw runs locally and publish
  only compact experiment records.
- Demonstrate a sourced end-to-end answer on a suitable development request, plus
  empty-result and upstream-error behavior. Keep held-out questions frozen and
  do not transfer historical qrels to newly fetched candidate sets. PMC full text
  remains an explicit experiment while the live default uses PubMed abstracts.

Semantic-support errors remain a release blocker. Reserve work on development
cases must precede final held-out confirmation if those errors remain material.
The API and following sessions shift by two sessions to make room for live search
and the separate broader-topic/manual-annotation stage.

## Native runtime optimization

The user requested performance work after accepting live search. The
[MLX runtime](native-mlx.md) targets Qwen load/prefill/decode, the dominant costs.
Fixed-source replays and fresh live requests measure latency and integrity;
they do not replace or prefill the user's independent annotations. Exact citation/source
gates and bounded output contracts remain in force. Current
question-driven biomedical prompts are versioned separately from the historical
4B v7 and 9B measurements. A runtime, prompt or quantization change is not evidence
of semantic equivalence.

## Annotation status and independent human-reference protocol

The current development path uses the separate completed assistant annotation
for the one-day exploratory assessment above. No assistant judgments are imported
as human labels. The protocol below preserves the earlier independent
human-reference requirements for any future human-validated evaluation; it does
not block the explicitly requested assistant-based development diagnostics.

The [integrity record](experiments/biomedical-scope-preparation.json) covers the
verified six-question, 84-card package, source bindings and automated functional
checks. The real annotation page remains blank. Visual browser QA was not completed
because local-file navigation was rejected; this remains a QA limitation, not a
quality acceptance or a substitute for user annotation. No Qwen answers were
generated for the six source-preparation cases.

The user placed broader English biomedical support after live search so a new
independent manual relevance-annotation round could precede quality measurement.
The frozen cases and operational workflow are documented in
[newbroader-biomedical.md](newbroader-biomedical.md). The
existing manual labels are retained for their original question/article pairs;
AI-generated qrels v2/v3 are separate historical assessment sets, not a substitute
for the user's new labels. Existing depression/anxiety development results do
not establish quality on other specialties.

The original independent-human protocol requires annotation first, quality
measurement afterward, with all judgments supplied independently by the user. Codex prepares the source package,
annotation interface/rubric and consistency checks, but supplies no prefilled
labels, suggested grades or model-written annotation rationales. Pending labels
are a gate, not permission to substitute AI judgments or publish interim quality
scores. Runtime and integrity checks may proceed independently.

Preparation and acceptance criteria:

- Freeze a manageable cross-topic development question set and the relevance
  intent for each question before tuning. Include several biomedical specialties,
  question types and population scopes. Preserve the existing held-out set.
- Fetch and freeze the article candidates and exact title/abstract versions. Build
  a reusable annotation package with question, article text and source links,
  consistent 0/1/2 relevance definitions, and an explicit unknown/unjudgeable option.
  Hide retrieval method, score and rank during labeling where practical.
- The user alone supplies every judgment and rationale. Store annotation provenance
  and bind every label to question ID, PMID and text version. Do not autofill missing
  labels with model assessments or treat unlabeled articles as irrelevant. Preserve
  the user's originals; flag inconsistencies for the user rather than rewriting
  their judgments.
- First deliver the blank package; then wait for the user to complete and explicitly
  finalize the required annotation round. Validate its structure and source bindings,
  freeze its version/checksums, and only then run quality measurements against it.
  Do not run scoring or use provisional labels to tune or select the pipeline.
- Keep retrieval relevance and answer correctness as distinct assessment targets.
  Article 0/1/2 relevance labels support retrieval metrics; they alone cannot label
  a generated claim as supported, complete or correct. Define the answer-evaluation
  rubric and required human reference evidence before measuring answer quality.
  If additional answer/claim judgments are needed after draft generation, the user
  also performs those independently, and answer metrics wait for their completion.
  Report judgment coverage and remaining uncertainties for both evaluation stages.
- Remove depression/anxiety-specific constraints from query construction, source
  selection, prompts and user-facing product descriptions. Audit assumed adult
  populations and treatment-efficacy framing: question scope must determine which
  population, intervention, comparator and outcome restrictions apply.
- Support question types beyond treatment efficacy where the available sources
  permit, including diagnosis, prevention, prognosis and biological mechanisms.
  Make evidence selection and review appropriate to the requested question rather
  than applying an efficacy-only interpretation to all questions. Preserve strict
  citation checks and the requirement to withhold unsupported statements; changes
  to review behavior require explicit development evidence, not an automatic
  switch to the previously rejected experimental reviewer.
- Use the new manually annotated development cases (for example cardiovascular
  disease, diabetes, infection and neurology) to inspect query construction,
  retrieval and source mappings. Keep historical benchmark scope labels and
  distinguish evidence of functionality from evidence of quality.

The expanded scope covers biomedical questions supported by PubMed literature,
not general web search. The new set is a development set unless a separate
held-out protocol is agreed; publication of labels does not make it independent
test data. Future independent human validation is separate from the one-day
assistant-based development assessment.

## Comparison rules

- Compare both MedCPT modes on the same PubMed snapshot and exact text windows.
  The Dense mode scores all windows of its ten selected articles; direct MedCPT
  scores all windows of all candidate articles. Article score is the maximum.
- Check MedCPT's entire pair budget, including the question, title and special
  tokens. If input constraints change window boundaries, rebuild the matching
  Dense artifacts. Keep historical scores separate from the new comparison.
- Include PubMed and Dense without MedCPT as controls. Existing Qwen reranking
  results are historical references, not measurements of the new protocol.
- Preserve the active evaluation policy and disclose missing judgments. Do not
  silently treat new unjudged results as irrelevant. Record annotation provenance;
  AI assessments are not independent expert judgments.
- Freeze held-out questions and the evaluation procedure before tuning the new
  variants. Use them for final confirmation, not repeated model selection.
- Choose a pipeline using quality, latency and memory together. Specialization
  alone does not establish MedCPT's superiority. Historical prose experiments used
  Qwen for generation and same-model review; the current MVP selects source
  quotations without claiming that separate factual-review stage.

## Release conditions

The [2026-11-16 amendment](usefulness-research.md#acceptance-criteria)
replaces the useful-selection floor with the best measured quality among the
sequentially tested justified hypotheses, reported with counts, denominators and
unknowns. It sets no minimum useful percentage or count for MVP readiness. The
separate unnecessary-abstention ceiling of 50% among reference-confirmed
answerable queries and complete-request maximum of 60 seconds remain in force.
Existing source-quality facets, critical scope-error and independent-reference
requirements are unchanged; removing the useful floor does not approve release.

Generation remains the largest uncertainty. Better retrieval does not establish
that a generated claim preserves age, subgroup or comparison qualifiers. Exact
citations and same-model review are not proof of support. If Days 20–21 do not
resolve material errors, allocate reserve sessions or narrow the answer format
before release; do not publish a readiness claim based on the day number.

Keep one supported MVP application pipeline on macOS/Apple Silicon with
Qwen3.5-4B through MLX and the existing BGE retrieval path. Cross-platform
runtime backends are post-MVP work, not a release condition. Git history
preserves removed experimental alternatives; concise experiment reports preserve
their measured results and the reasons for rejecting them. Delete obsolete code only after comparison
and final checks. Model weights, source corpora, vectors and raw run outputs
remain local artifacts rather than Git contents.

PMC scope and the Medium profile must remain within measured laptop limits.
The 2026-11-11 broader-topic stage separates code/source/package preparation,
assistant-based development diagnostics and any later independent human
validation. It supersedes the original post-release expansion restriction.
Broader topic support still requires its own retrieval and answer assessment;
depression/anxiety benchmark scores must not be generalized to other specialties.
The six-day schedule does not itself establish clinical reliability or satisfy
missing independent human-reference requirements.
