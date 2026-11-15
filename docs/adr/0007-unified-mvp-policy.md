# Unify the supported MVP policy

Status: selected for final validation on 2026-11-15, before held-out execution.

## Decision

Ship one macOS/Apple Silicon application policy: fresh bounded PubMed abstracts
→ BGE/MPS Dense ranking → exact evidence → Qwen3.5-4B/MLX source-findings cards.
The primary interface is the simple `medlens` terminal session. `medlens-api`
provides the same policy on loopback HTTP. Default acquisition/display limits
remain 100 candidates and ten ranked articles, with the existing bounded
request parameters. There is no additional reranker, automatic runtime fallback
or stale-source substitution.

The application selects `source-findings` exclusively. Remove the public CLI
answer-style selector. API requests may omit `answer_style`; when supplied it
must be the constant `source-findings`. Reject `draft` rather than silently
changing or accepting its meaning. Keep diagnostic search-only operation.

This supersedes the public prose-draft default and primary-entry-point portions
of [ADR 0003](0003-primary-pipeline.md),
[ADR 0005](0005-qwen4b-development-default.md) and
[ADR 0006](0006-biomedical-scope-and-human-reference.md). Their model provenance,
source bindings, historical measurements and independent-reference requirements
are retained. The pinned 4B weights/tokenizer, MLX environment and BGE policy
are unchanged. [The supported MVP](../mvp.md) is the current operational index.

## Why

Development evidence found material semantic errors in generated prose and
same-model review. The narrower exact-quotation format improved the repeatedly
tuned six-case development assessment, while leaving selection omissions and
uncertain applicability unresolved. That AI assessment supports choosing a
bounded research display; it does not prove unseen-question reliability or
clinical validity. No held-out result selects or tunes this policy.

Source-findings selects study-context and finding quotations. It does not
generate a factual synthesis or claim that a separate model factual-review
stage occurred. Nonempty results remain `partial`; exact quote identity is
distinct from support for the question, completeness and independent validation.
Invalid output is withheld, and applicability cautions remain visible.

## Consequences

- The CLI, API, default asset setup and current guides describe one policy.
  The main environment uses `requirements-mvp-macos-arm64.lock`; the isolated
  MLX lock remains separate. Optional HTTP dependencies use `requirements-api.lock`.
- Default asset preparation verifies/downloads BGE and its tokenizer only;
  Qwen4B preparation is separate. Historical 9B, MedCPT and Metal-kernel assets
  require the explicit research asset profile and are not MVP prerequisites.
- Research/replay scripts and frozen experiment records retain explicit older
  choices. Prose drafts require `--answer-style draft` in those research tools;
  9B, Transformers and reranking are not application alternatives.
- English biomedical input remains the supported scope. The lexical planner
  does not classify language/topic or translate; no ASCII-only rule is added.
  Exact PubMed overrides and the unchanged original question remain inspectable.
- Windows/Linux, CUDA, CPU-only Qwen, alternative models/backends, PMC full text,
  Medium-scale expansion and a web UI stay outside the MVP.
- Functional checks and exact citations cannot close semantic-quality or
  independent-human-reference gaps. Final validation must disclose them,
  preserve original labels and distinguish missing judgments from negative
  judgments. Known defects are not erased by this policy decision.

Final validation, dependency/CI checks and a full clean-model installation remain
separate release evidence. This decision is a configuration freeze, not a
release-readiness declaration.
