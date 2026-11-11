# Broaden biomedical requests and prepare an independent human reference

Status: accepted for development on 2026-11-11. Functionality and source integrity
can be checked now; cross-topic retrieval and answer quality await the owner's
independently completed and explicitly finalized manual references.

## Decision

Accept English biomedical research questions without a depression/anxiety topic
allowlist. Keep the default Qwen3.5-4B model, bounded fresh PubMed acquisition,
BGE Dense ranking, exact evidence catalogs and isolated native runtime from
[ADR 0005](0005-qwen4b-development-default.md). Keep original source snapshots,
article/window/vector bindings and the existing token/claim limits.

PubMed receives a transparent deterministic search expression derived from the
question's wording, or the exact explicit `--pubmed-query` override. Changes to
the submitted expression must be recorded; the complete original question still
serves Dense and Qwen. This is candidate acquisition, not a validated clinical
query compiler. No model call or disease dictionary is added to query planning.
PubMed's query translation and every source acquisition remain inspectable.

Use `cited-answer-v8` and `axes-first-v2` in the default compact policy. Requested
population, species/system, comparison and finding target determine the review;
no adult population, treatment improvement or placebo control is assumed when
unasked. Preserve source restrictions and do not extrapolate between human,
animal and cell findings. Diagnostic accuracy, prognosis and mechanisms need
source support for their own targets. Three named v4 axes and an explicit verdict
remain internally constrained and strictly parsed. The rejected experimental v5
reviewer is not selected. Historical explicit 9B comparisons keep their legacy
protocol and are not a separately validated general-biomedical route.

Evidence selection already uses generic article diversity, Dense window scores
and results/conclusion section preference. It contains no psychiatric disease
filter or requirement for an adult treatment trial; this change does not select
new passages using human labels or claim an improvement to that policy.

## Human reference before quality measurement

Freeze six cross-topic **development** questions, their relevance intent and
annotation rubric before preparing the final source pool. Preserve the original
plan bytes and source-acquisition attempts. Freeze title/abstract text versions
and a blind union of PubMed and Dense top-ten results from each same candidate
snapshot. Do not choose or discard articles using assistant relevance judgments.

Deliver a blank local annotation interface. The owner independently supplies
every relevance grade and rationale; there are no default grades, model-written
rationales or suggested answers. An untouched item differs from an explicit
unjudgeable decision. The finalized export binds the package, query, PMID and
text version; preserve its original bytes and SHA256 in a fresh reference folder.

No new quality metric, tuning from provisional labels, or AI replacement for
missing judgments is allowed before explicit human finalization. Unknowns remain
unknown and judgment coverage remains visible. The existing held-out set and
historical qrels are not used by this preparation workflow.

Article relevance labels assess retrieval. They cannot establish claim support,
qualifier preservation, answer completeness or usability. Answer-quality
measurement requires a separate frozen output/reference protocol and another
blank human form completed by the owner. Those criteria are declared in the
[development plan](../../benchmarks/biomedical-dev-v1/plan.json), before answer
scoring. Runtime and exact-source integrity checks are distinct from quality
estimates and may proceed without semantic labels.

## Consequences

This expands supported input functionality, not a claim of clinical reliability
or quality across specialties. Existing psychiatric results and the prior small
4B human comparison remain historical evidence for their original questions and
outputs. They are not transferred to changed prompts or newly fetched articles.

See the [workflow guide](../newbroader-biomedical.md) for reproducible preparation,
manual finalization and known query-planning limits. Raw source texts, vectors,
annotation packets and model outputs stay local. Only the plan, tooling,
protocol and compact integrity/runtime evidence are versioned at this stage.
