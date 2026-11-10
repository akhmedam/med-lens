# Use Qwen3.5-4B as the development answer default

Status: selected by the user on 2026-11-10 for development. This supersedes the
9B model/default choice in [ADR 0003](0003-primary-pipeline.md) and the 9B-only
profile in [ADR 0004](0004-native-mlx-runtime.md). Their Dense/source-selection
architecture and isolated native runtime remain in force.

## Decision

Default `ask_live.py` and `ask.py --generate` to native MLX Qwen3.5-4B. Keep
`--answer-model 9b` as an explicit comparison and require it with
`--engine transformers`. The primary workflow remains fresh bounded PubMed
acquisition, BGE Dense ranking, exact source selection, draft generation and
same-instance support review. There is no additional primary reranker or silent
fallback to another model.

Use `mlx-community/Qwen3.5-4B-MLX-4bit` revision
`32f3e8ecf65426fc3306969496342d504bfa13f3`, affine 4-bit/group-64, including the tied
embedding/head. Preserve its BF16 remainder and FP32 `A_log`; do not label it as
the earlier FP16 9B conversion. The community card does not disclose an exact
upstream official weight revision or converter commit. Pin the official 4B
tokenizer separately to `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

The `compact-v1` answer policy uses `cited-answer-v7` short draft instructions and
the unchanged exact quote catalog. Its `axes-first-v1` reviewer emits six named
v4 fields: claim ID, population, comparison, outcome, explicit verdict and reason.
Positional arrays are not the default. Decoder constraints and strict post-parse
checks reject contradictory combinations without filling fields, repairing JSON
or inferring support from passing axes. Explicit v5 review is unsupported by
this compact policy and fails clearly; the legacy 9B route remains separate.

Preserve maximum three claims, one to three exact citations per claim, 4,000
evidence tokens, 6,500 full input tokens, and 768/512 output-token caps. Keep the
finite-score guard, complete JSON/citation validation, fail-closed output limits
and source-preserving failure reports. Cached compact calls must match model,
revision, artifact, answer policy and decoder contract.

## Evidence and limits

The user independently filled and finalized the new blind reference: 30 unique
claims, 32 occurrences, six related source contexts, two independent development
questions and one annotator. The assistant checked artifact bindings and counted
the supplied human labels; no AI semantic labels or relevance qrels were used.
See the [human assessment](../qwen4b-manual-human-assessment.md) and its
[aggregate record](../experiments/qwen4b-manual-human-assessment.json).

On that common pool, the named 4B reviewer retained 16/24 human-eligible claims;
9B retained 15/24. However, 4B accepted 3/5 human comparison-mismatched claims plus
one uncertain claim; 9B accepted 4/5 mismatches. Both failed the predeclared
zero-known-bad-acceptance gate. The mismatches must not be relabeled as factual
fabrication. Logical decoder consistency does not establish scientific support.

The user rated the recombined 4B answers usable in 5/6 contexts and complete in
2/6, versus 4/6 and 0/6 for 9B. The all-context 4B usability gate still failed.
The absence of a paired completeness loss is limited by having zero adequate
baseline answers. These results do not establish clinical reliability, general
biomedical coverage or model equivalence. A larger model is not a validated
automatic remedy for the observed errors.

The user selected 4B as the faster development default under these known limits.
Historical paired reviewer timing observed a 2.27x median call-time ratio; that
does not include draft generation, startup, grammar compilation, retrieval or
PubMed. Current integrated measurements must retain their own scope and cannot
inherit the earlier 9B runtime observations. The current
[integration record](../experiments/qwen4b-default-integration.json) covers one
fixed-source smoke and two fresh 100-candidate live requests. It verifies pipeline
completion, runtime closure and source binding; changing live evidence prevents
a paired speed comparison, and these outputs received no new human quality labels.
See the [native guide](../native-mlx.md).

## Consequences

Setup can fetch and validate 4B plus its official tokenizer without restoring 9B
weights: `scripts/prepare_mlx.py --download`. Native 9B conversion is explicit via
`--model 9b`. Model/profile identity, pinned sources, grammar policy, prompt and
actual input-token hashes remain auditable. The user's exact finalized human
exports are versioned under `data/evaluation/qwen4b-manual-human-v1`, explicitly
marked as human annotations and containing labels, bindings and rationale rather
than article/model text. Raw source/article text, model outputs and weights stay
local. The machine-readable [decision record](0005-qwen4b-development-default.json)
contains aggregate evidence and provenance identifiers.

Displayed answers remain research drafts requiring human review. Same-model
support review is a heuristic, not independent verification. This decision
changes the development default; it does not approve unattended final answers.
Existing frozen 9B experiment commands, including `scripts/generate_answers.py`,
retain their historical model/protocol instead of silently changing meaning.
