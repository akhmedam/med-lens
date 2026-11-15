# Prospective human confirmation

This is a new **test/confirmation** round, separate from the six biomedical
development cases and the twelve already observed held-out questions. Its
purpose is independent manual human assessment of the selected application,
with quality and complete-request latency checked on the same future live
outputs. It is not a new development set or permission to tune on new labels.

## Prepared first-round form

The [frozen human-confirmation-v1 plan](../benchmarks/human-confirmation-v1/plan.json)
contains twenty new questions. The current V2 source-reference package contains
**208 article cards plus twenty query-answerability decisions**, built without
Qwen answers, human labels or quality scores. The
[preparation record](experiments/human-confirmation-preparation.json) binds this
package and its source identities. Open the delivered standalone copy:

```sh
open ~/Downloads/MedLens-human-confirmation-v1.html
```

Read each saved article and choose `0`, `1`, `2` or «Не могу оценить». For each
question choose «Да», «Нет» or «Не могу оценить» for source sufficiency. When you
choose «Да», tick at least one supporting PMID in the article cards. Comments
are optional. After completing all questions and articles, click «Зафиксировать
и скачать JSON» and return that original downloaded JSON without editing it.

The HTML alone is sufficient on another computer; browser decisions are not
written back into the HTML file. Export JSON before changing browser or computer.
The complete original package remains at
`results/human-confirmation-v1/manual-source-reference-20261115-v2` for validation.
This is only the first source/answerability round: actual future live outputs
require a separate human usefulness and card-support review before scoring.

## Owner's acceptance criteria

The owner specified the following thresholds before this new round:

- At least **50% of all planned queries** must produce a useful selection.
- Unnecessary abstentions must be **at most 50% of human-confirmed answerable
  questions**: `false_abstention_count / answerable_yes_count <= 0.5`. For `A`
  confirmed answerable questions, no more than `floor(A / 2)` false abstentions
  are permitted. A zero denominator is not an automatic pass.
- Every accepted request must finish within **60 seconds from beginning to end**.

The existing release blocker for materially wrong population/system,
intervention/comparison or outcome remains in force. Fifty percent useful
selections does not authorize wrong cards in the remaining selections. Exact
source quotations and a `partial` status do not establish scientific support.
Source-integrity, privacy and runtime-cleanup requirements also remain separate
acceptance conditions.

The primary confirmation round contains **20 planned questions** using default
100-candidate acquisition and top-10 ranking. With complete human decisions,
at least ten queries must be useful. All twenty remain in the denominator;
failed requests, timeouts, empty searches and inconvenient outcomes are not
silently excluded or replaced.

## Freeze questions and prepare sources first

Before any source acquisition, freeze the question texts, their scope/intents,
specialty and question-type composition, pool rules, acceptance criteria and
source-preparation configuration. Record the plan's byte SHA256. Do not edit
this round in place after inspecting search yield or human judgments. Do not use preparation
results from these new cases to tune query planning, retrieval, prompts or
selection; defects are repaired on the old development material.

The first preparation phase performs **search only**. Each question obtains a
fresh bounded PubMed title/abstract snapshot and the pinned BGE ranking. No Qwen
answer is generated, no article is assigned a relevance label and no quality
score is computed. Preserve unsuccessful and empty cases as well as successful
ones. A failed preparation attempt is retained; any explicit infrastructure
retry needs its own recorded reason and attempt identity, without selecting a
more convenient question or hiding earlier evidence.

The manual reference pool is the deduplicated union of the first ten PubMed
Best Match articles and first ten Dense articles from the same source snapshot.
There are at most twenty article cards per query, at most four hundred for this
round. This is a bounded reference pool, not exhaustive PubMed coverage.
Keep full saved title/abstract text, exact PMID and text-version identity, raw
source bindings and the acquisition/ranking configuration. Hide method, score
and rank in the article annotation view; fix its blind order deterministically.

The delivered `index.html` must contain its question, article, rubric and form
data and work without neighbouring files or network requests. Opening a PubMed
link is optional external navigation. Every decision starts pending. Browser
storage is a convenience; the owner's exported JSON is the portable record.

## First manual reference: articles and answerability

Only the owner supplies the following decisions and any rationale:

| Field | Allowed decisions | Meaning |
| --- | --- | --- |
| Article relevance | `0`, `1`, `2`, explicit unknown | No useful evidence; useful partial evidence; directly addresses the requested target and restrictions; unable to judge. A supported null finding can be directly relevant. |
| Query answerability | Yes, no, explicit unknown | Whether the provided frozen sources contain enough supported information for at least one useful source finding addressing the requested question. |
| Supporting sources | Exact article/item IDs, optional human rationale | Evidence for the owner's answerability decision; never an assistant suggestion. |

Answerability is relative to the supplied source pool, not a claim that the
question is answerable or unanswerable throughout biomedical literature. A
missing abstract detail is unknown, not positive evidence of the requested
population or comparison. An empty preparation result supplies no automatic
human label. Pending, explicit unknown and a negative judgment remain distinct.

Do not prefill grades, answerability, expected answers, suggested rationales or
AI semantic judgments. The assistant validates format and bindings, not the
scientific correctness of the owner's values. Finalization requires an explicit
decision or explicit unknown for every required item; pending must be zero.

## Verify and preserve the owner's export

The package is already built; these commands do not repeat acquisition or
regenerate it. Run them from the checkout with the complete V2 package. Replace
the example export path with the unchanged JSON actually downloaded by the owner;
choose a fresh freeze-output directory.

```sh
MEDLENS_CONFIRMATION_PACKAGE=results/human-confirmation-v1/manual-source-reference-20261115-v2
MEDLENS_HUMAN_EXPORT=/path/to/your-finalized-export.json
.venv/bin/python scripts/prepare_confirmation_annotation.py verify \
  --package "$MEDLENS_CONFIRMATION_PACKAGE"
.venv/bin/python scripts/prepare_confirmation_annotation.py validate \
  --package "$MEDLENS_CONFIRMATION_PACKAGE" --human-export "$MEDLENS_HUMAN_EXPORT" --require-finalized
.venv/bin/python scripts/prepare_confirmation_annotation.py freeze \
  --package "$MEDLENS_CONFIRMATION_PACKAGE" --human-export "$MEDLENS_HUMAN_EXPORT" \
  --output results/human-confirmation-v1/finalized-human-source-reference-001
```

Verification checks the frozen sources and form. Validation checks the human
export's structure, completeness and exact bindings without displaying semantic
grades or rationales. Freeze preserves its original bytes read-only with a
SHA256 receipt and explicit **MANUAL HUMAN ANNOTATION** provenance. These tools
compute no usefulness, abstention or other quality score.

## Quarantine until the candidate is frozen

Preserve every original human export byte for byte in a fresh read-only
reference directory, with its SHA256, package/plan/source identities, declared
manual-human origin and validation receipt. Never rewrite a human value to
make a package, metric or result convenient.

Until the candidate application policy is frozen, automated validation may
check checksums, schemas, item/source identity, finalization and completion
counts. It must not expose per-item grades, answerability decisions, rationales,
grade histograms or quality metrics to the people or agents repairing the
pipeline. Those repairs use the existing development material and previously
observed test material. If new judgments guide tuning, explicitly reclassify
this round as development and prepare another untouched confirmation round.

The candidate freeze records source and configuration identities, model and
tokenizer revisions/checksums, prompt/selector policy, runtime and request
limits, query planning, cache policy and the planned performance-stress matrix.
Only then begin its one live confirmation pass. Human reference values are
scoring-only inputs: never use them to filter articles, select evidence, modify
queries or populate the model prompt during that pass.

## One future live pass and source reconciliation

Run every planned question once through the complete selected **live** path.
Preserve fresh snapshots, actual candidate pools, public/private results,
source identities, timings and cleanup observations. Do not select a better
retry or change the prompt, model, limit or question after seeing the result.
Existing declared bounded transport retries remain part of the request and
its elapsed time. An interrupted or failed attempt remains recorded.

Future PubMed sources may differ from the preparation snapshot. Reuse a human
article decision only when the question, PMID, exact title/abstract content and
text-version identity match. Produce an explicit old-to-new mapping receipt;
the same PMID alone is insufficient. New or changed items start blank in a
delta package. A changed reviewed reference pool requires the owner to
reconfirm that query's answerability, even if some individual labels survive.
Here, a pool change means membership or source text/version changes in the
reviewed reference union, not merely a different download timestamp. Delta
source/answerability forms keep the generated answer and model abstention hidden;
only after that reference is finalized does the owner see the output-review form.
Do not silently map missing coverage to irrelevant or answerable labels.

Reconcile all actual sources before quality scoring. If required judgments
remain missing or unknown, preserve that uncertainty and report acceptance as
undetermined when it can change the threshold result. No frozen-replay timing
plus a separately measured download time substitutes for the complete live
request measurement.

## Second manual reference: the actual outputs

The owner reviews the actual frozen live outputs against their exact sources.
The second form is distinct from article relevance and answerability. It
contains no AI verdicts and no filled decisions or rationales.

- For each visible card: factual/source support, requested population/system,
  requested intervention/comparison, outcome/target and preserved qualifiers.
- For each query: useful selection, completeness/applicability observations
  and any explicit critical scope mismatch, with yes/no/unknown decisions.
- For an abstention: whether sufficient provided evidence existed, with the
  reconfirmed answerability reference and an explicit human decision.

A useful selection contains at least one supported, applicable finding that
helps answer the question, without a materially misleading displayed finding.
It need not be a comprehensive synthesis; a supported null or negative result
can be useful. Merely mentioning the topic, copying a sentence exactly or
showing a correct paper title is insufficient.

An unnecessary abstention is an `insufficient_evidence` result despite human-
confirmed sufficient provided evidence. Provider/model errors, invalid output,
cleanup failure and timeouts remain operational failures, not successful
semantic abstentions. Missing judgments are neither useful selections nor
verified appropriate refusals.

Freeze the explicitly finalized second human export without changing its
values. Only afterward compute the predeclared quality metrics and their
coverage. Store `MANUAL HUMAN ANNOTATION` provenance explicitly. AI diagnostics,
if separately requested, remain distinct and cannot replace this reference.

## Complete-request time and supported parameters

Measure from the CLI submission action/Enter or client POST start to the fully
available final result after actual model cleanup. Include validation, admission,
network/backoff, parsing, new embeddings, model loading, generation, persistence,
serialization and terminal/client delivery. Immediate job acceptance is not the
finish time. Installation of assets before requests and application startup
before the submission action are separate setup observations.

The limit is **60 seconds for each request**, not the median, an average, a
percentile or a sum of independently timed stages. Record elapsed time for all
outcomes. A quick failure cannot be called a useful answer. Any observed request
over sixty seconds fails its timing requirement; do not introduce an exception
after inspecting it.

The requirement applies to all accepted application settings, including
candidate limits 1–300 and top-k limits 1–10. The twenty default-profile queries
alone do not establish coverage of that configuration space. Freeze a separate
development stress matrix for boundary settings, varied source sizes and
declared cold/warm passage-cache states before running it. Such stress cases
must not be used to choose a better output for a confirmation query.

The current service does not implement a hard sixty-second cancellation budget.
Deadline enforcement and safe cleanup remain engineering work; this document
records the requirement and does not claim an already enforced guarantee.
Observe current parent/child RSS separately, disclose validation-harness
overhead and confirm native PID cleanup; do not add overlapping RSS/Metal
counters or infer long-session leak freedom from short observations.

## Report the result without relabeling uncertainty

Report fixed query counts, human-reference coverage, exact useful-selection and
unnecessary-abstention numerators/denominators, per-request elapsed time, all
failures and critical mismatches. Explicit unknowns remain unknown. If unknown
decisions could move a metric across its threshold, the gate is not established.
Usefulness is `useful_yes_count / 20`; explicit unknowns contribute neither a
positive nor an imputed negative judgment. Keep the fixed denominator and report
bounds when unknowns could affect acceptance. For unnecessary abstentions, the
denominator is the reconfirmed `answerable_yes_count`, not all model refusals.
Unknown answerability and unknown usefulness stay distinct. If a denominator is
zero, its rate is undefined and the gate is `not_established`, not automatically
good. Acceptance uses the owner's predeclared sample-rate thresholds; do not
silently substitute a confidence-bound threshold after the results are seen.

These are sample-level acceptance criteria for this explicitly scoped MVP.
Twenty queries and one owner's judgments do not establish general biomedical or
clinical validity. After these outputs or labels inform a behavior change, a
later confirmation requires a new untouched version. Preserve the original
round and every source/result/reference binding.
