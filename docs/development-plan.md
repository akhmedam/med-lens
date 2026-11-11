# Development plan

The current development pipeline is live bounded PubMed acquisition → BGE Dense
ranking → exact evidence → native MLX Qwen3.5-4B draft and same-model support
review. The [4B development-default decision](adr/0005-qwen4b-development-default.md)
supersedes the model choice in ADR 0003. The current broader biomedical prompts
are `cited-answer-v8` and named `axes-first-v2` under the unchanged six-field v4
taxonomy. Earlier 9B and 4B v7 measurements retain their original scope; they do
not establish broader-topic quality or approve autonomous answers.

The user requested annotation first and will personally supply all new judgments.
The [broader biomedical stage](newbroader-biomedical.md) prepares six frozen English
cross-topic development cases, source snapshots and a blank human relevance
package. Preparation is complete: six search-only runs froze 246 candidate records
and a verified 84-card blank package; all human decisions remain pending. See the
[preparation/integrity record](experiments/biomedical-scope-preparation.json).
Human article annotation and explicit finalization gate retrieval metrics.
A separate human claim/answer reference gates later answer-quality metrics. No
provisional labels, AI-written rationales or historical qrels substitute for either.
The existing held-out set remains untouched.
[ADR 0006](adr/0006-biomedical-scope-and-human-reference.md) records the new scope
and separate human-reference gates.

The remaining schedule below preserves the historical session numbering; it is
not a calendar deadline or a readiness criterion. API and interactive CLI work
follow source/annotation preparation. Human quality checks and unresolved semantic
support errors can require reserve work before a portfolio release. The earlier
Air recovery, live-search, PMC and ranking records remain useful functional and
historical evidence within their documented limits.

The primary user interface will be an interactive command-line application.
HTML reports remain an optional export. The API is a separate programmatic
interface; a web UI is outside this release plan.

## Remaining sessions

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
| 25 | Prepare broader English biomedical question handling and a blank source-bound manual annotation package. The user completes all judgments independently before quality evaluation begins. | [Frozen six-case plan and verified 84-card blank package](newbroader-biomedical.md) completed; independent human annotation remains pending. Retrieval and answer quality remain unmeasured until their separate required human references are finalized. |
| 26 | Add a minimal programmatic API with request parameters, errors and model lifecycle management. | The integrated live-search pipeline is callable through an API. |
| 27 | Build the interactive CLI: question input, mode selection, progress, ranked articles, evidence, drafts/citations, errors and report export. | A usable terminal workflow without a web UI. |
| 28 | Evaluate the frozen held-out questions, unsupported requests, empty evidence and repeated runs. | A final report on questions not used to tune the pipeline. |
| 29 | Remove losing implementations and unused dependencies; finish necessary CI checks, README and experiment documentation. | One maintained pipeline with a traceable explanation of its selection. |
| 30 | Follow setup from a clean checkout; prepare examples, limitations and the release. | A reproducible portfolio v1.0, subject to the quality checks below. |

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

## Completed preparation; independent manual annotation pending

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

The user explicitly requires annotation first, quality measurement afterward,
and will perform all annotation independently. Codex prepares the source package,
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
not general web search. The new manual set is a development set unless a separate
held-out protocol is agreed; publication of labels does not make it independent
test data. Annotation availability may require reserve sessions beyond the nominal
schedule.

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
  alone does not establish MedCPT's superiority. Qwen remains the answer generator
  and current support reviewer, with the documented limits of same-model review.

## Release conditions

Generation remains the largest uncertainty. Better retrieval does not establish
that a generated claim preserves age, subgroup or comparison qualifiers. Exact
citations and same-model review are not proof of support. If Days 20–21 do not
resolve material errors, allocate reserve sessions or narrow the answer format
before release; do not publish a readiness claim based on the day number.

Keep one supported runtime pipeline after selection. Git history preserves
removed alternatives; concise experiment reports preserve their measured results
and the reasons for rejecting them. Delete obsolete code only after comparison
and final checks. Model weights, source corpora, vectors and raw run outputs
remain local artifacts rather than Git contents.

PMC scope and the Medium profile must remain within measured laptop limits.
The 2026-11-11 broader-topic stage separates code/source/package preparation
from the pending human annotation and its later retrieval and answer evaluations.
It supersedes the original post-release expansion restriction. Broader topic
support still requires its own retrieval and answer evaluation; depression/anxiety benchmark scores must not be generalized to other
specialties.
