# Development plan

MedLens supports one local application workflow on macOS with Apple Silicon:
fresh PubMed titles and abstracts → BGE/MPS Dense ranking → exact evidence →
Qwen3.5-4B/MLX source findings. The plain-text CLI and loopback API share the
[same application policy](adr/0007-unified-mvp-policy.md). Nonempty findings
remain partial and independently unverified.

## Current state

- Fresh PubMed acquisition, reusable vectors, source-bound quotations, the
  [interactive CLI](interactive-cli.md) and [local API](local-api.md) are
  implemented. Each accepted request owns an isolated process group.
- [Request isolation and limits](request-reliability.md) enforce a shared budget:
  55 seconds for work, three for confirmed cleanup and two reserved for delivery.
  Timeouts are failed incomplete requests, not evidence abstentions.
- The [sequential quality study](usefulness-research.md) selected H6 after testing
  justified hypotheses on observed development material. Its newly frozen live
  sample produced 6 useful selections and 1 uncertain selection out of 20,
  5/16 unnecessary abstentions (uncertainty bounds 31.25–35.29%), and four
  confirmed comparison/outcome substitutions. All twenty requests completed
  within sixty seconds; maximum complete-request time was 41.44 seconds.
- Semantic-support defects remain release blockers. Passing operational,
  quotation-integrity and cleanup checks does not establish scientific validity.
  The latest assessment is explicitly AI-origin, not independent human review.

## Milestones

### Code, CI and documentation

- [x] Consolidate installation into required runtime dependencies and a separate
  development toolchain. Keep the isolated MLX environment required because its
  dependency versions differ from the main environment.
- [x] Replace development-session filenames and headings with descriptive names.
  Preserve historical experiment payloads, source identities, hashes and genuine
  manual-human annotations.
- [x] Configure CI for offline regression, lint, package build and installed-wheel
  checks. CPU grammar fixtures are required development checks; model execution
  requires the isolated MLX environment and Apple hardware.
- [x] Keep the README focused on purpose, prerequisites, installation and the
  request flow; link detailed setup, usage and limitations from it.
- [x] Verify offline tests, dependency pins, lint and distribution packaging
  locally. Hosted CI runs when the changes are published. A change to selection,
  generation, timing or process cleanup reopens application acceptance.

### Clean installation and release candidate

- [ ] Follow the documented setup from a clean checkout on Apple Silicon.
- [ ] Verify pinned BGE and native Qwen assets, their source revisions and hashes.
- [ ] Exercise CLI and API through fresh retrieval, findings, delivery and cleanup.
- [ ] Verify explicit errors, cancellation, cold/warm cache states and accepted
  candidate/top-k boundaries on separate development inputs.
- [ ] Prepare versioned artifacts and release notes only after the installation,
  source-integrity, resource and quality conditions below are satisfied. Keep a
  release candidate with explicit blockers while those conditions remain open.

### Semantic quality and renewed confirmation

- [ ] Address remaining population/comparison/outcome substitutions using observed
  development material. Preserve study setup, null results and uncertainty.
- [ ] Freeze the candidate before a new confirmation round. Previously inspected
  cases and their labels are now development material, not untouched tests.
- [ ] Reconcile exact question/PMID/text versions for fresh PubMed sources and
  renew answerability judgments when the reviewed source pool changes.
- [ ] Review actual displayed outputs and abstentions separately from retrieval
  relevance. Disclose annotation origin and keep all failed attempts and unknown
  decisions in the reported denominators.

## Acceptance conditions

The [quality-policy amendment](usefulness-research.md#acceptance-criteria) sets
quality to the best measured result among sequentially tested justified
hypotheses. Report useful-selection counts, denominators, explicit unknowns,
regressions, null effects and failures. There is no minimum useful-selection
percentage or count for readiness.

The separate conditions remain at most 50% unnecessary abstentions among
reference-confirmed answerable queries and at most sixty seconds for each
complete accepted request, including delivery and confirmed cleanup. Critical
scope errors, source-integrity defects and unresolved resource issues remain
blockers. Exact quotations and same-model judgments do not prove semantic
support; independent human or clinical validation must never be inferred from
functional checks or AI annotations.

The original [confirmation plan](../benchmarks/human-confirmation-v1/plan.json)
and [protocol](human-confirmation.md) preserve their historical criteria. Genuine
manual-human annotations remain unchanged. Assistant assessments have separate
provenance and never fill pending human-reference fields.

## Platform and post-MVP work

The supported installation targets Python 3.11 on macOS/Apple Silicon, BGE/MPS
and Qwen3.5-4B through MLX. Windows/Linux, NVIDIA/CUDA, CPU-only Qwen execution
and automatic platform selection remain post-MVP work. Verify each platform
before claiming support.

Other deferred work includes a web UI, larger corpora and broader PMC use.
[PMC feasibility](pmc-fulltext.md) and [shared-corpus measurements](shared-corpus.md)
remain bounded experiments, not established relevance or scale improvements.
Alternative rankers, prose generation and the 9B comparison remain research
utilities with their documented limits.

## Evidence and decision history

Historical reports retain their original results and scope:

- [Primary-pipeline decision](adr/0003-primary-pipeline.md),
  [retrieval costs](retrieval-cost.md) and [transfer/recovery](handoff.md).
- [Generation format reliability](generation-format-reliability.md) and
  [semantic-support diagnostics](semantic-support-diagnostics.md).
- [Live search](live-search.md), [native MLX](native-mlx.md) and
  [broader-topic preparation](newbroader-biomedical.md).
- [Exploratory biomedical quality](biomedical-quality.md) and
  [source-findings assessment](biomedical-quality-improvement.md).
- [Earlier final operational validation](final-validation.md),
  [request-quality validation](request-quality-validation.md) and the later
  [sequential usefulness study](usefulness-research.md).

Corpora, weights, vectors and raw runs remain local artifacts. Git contains
code, compact reports and deliberately versioned labels; it cannot reconstruct
lost historical source snapshots from fresh downloads.
