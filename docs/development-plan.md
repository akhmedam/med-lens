# Development plan

Accepted on 2026-10-31. This replaces the earlier remaining-work schedule.
Days 1–15 are complete; the next session is Day 16. Day numbers denote working
sessions, not calendar deadlines. Target: a portfolio v1.0 at Day 28, with two
to four reserve sessions for generation, PMC and corpus scaling.

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
| 19 | Select the primary search pipeline from the measured trade-offs, record the decision and integrate it into requests. | One primary search path; Qwen reranking retained only as a historical experiment. |
| 20 | Improve generation format reliability: bounded claims/citations and structured output; measure refusals against Day 13. | Fewer format failures without weakening quote or output validation. |
| 21 | Evaluate population/age, severity, intervention, comparator, outcome, null findings and contradictions. | Generation checks and an explicit record of remaining semantic errors. |
| 22 | Pool local collections, handle duplicate records and text versions, verify cache updates and start the Small → Medium transition. | A shared collection with measured preparation and search costs. |
| 23 | Test a small permitted PMC full-text subset, section/provenance preservation and date filters; compare abstracts with full text. | A measured decision on full-text value and cost. Publication recency is not a proxy for evidence quality. |
| 24 | Add a minimal programmatic API with request parameters, errors and model lifecycle management. | The integrated pipeline is callable through an API. |
| 25 | Build the interactive CLI: question input, mode selection, progress, ranked articles, evidence, drafts/citations, errors and report export. | A usable terminal workflow without a web UI. |
| 26 | Evaluate the frozen held-out questions, unsupported requests, empty evidence and repeated runs. | A final report on questions not used to tune the pipeline. |
| 27 | Remove losing implementations and unused dependencies; finish necessary CI checks, README and experiment documentation. | One maintained pipeline with a traceable explanation of its selection. |
| 28 | Follow setup from a clean checkout; prepare examples, limitations and the release. | A reproducible portfolio v1.0, subject to the quality checks below. |

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
Expansion to other biomedical areas follows the stable initial release and
requires its own retrieval and answer evaluation.
