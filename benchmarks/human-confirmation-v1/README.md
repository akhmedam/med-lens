# Prospective manual-human confirmation v1

Twenty new English biomedical questions were frozen on 2026-11-15 before source
acquisition or annotation. The plan covers ten specialties (two cases each) and
five question types (four cases each). Questions/intents were assistant-authored;
all relevance, answerability and later output judgments belong to the owner.
No model judgments or expected answers are prefilled.

The owner selected acceptance criteria before this round:

- Confirmed useful selections / all 20 questions >= 50% (at least ten).
- Unnecessary abstentions / manually confirmed answerable questions <= 50%.
  A zero answerable denominator cannot pass. Unknown and missing decisions stay
  separate; do not convert them to positive or negative labels.
- Every full request <= 60 seconds from submission through result delivery and
  runtime cleanup. This is a strict maximum, not an average or percentile.
- Existing source-integrity and critical scope-error release gates remain in force.

The first source-reference round is search-only, with 100 candidates and top-ten
PubMed/Dense union pooling. Empty cases are retained, not replaced. The original
plan bytes and SHA256 are immutable. Source text/HTML/labels stay local, outside Git.

See [the protocol](../../docs/human-confirmation.md) for two-stage human annotation,
blind ordering, quarantine before the future candidate freeze, and fresh-source
delta handling. The future actual live outputs supply the usefulness/abstention
reference; today's article labels alone cannot score those metrics. A frozen-source
replay plus an old download time is not an end-to-end latency measurement.

Repairs use the old development material and the already consumed original
heldout-v1. Do not tune on these new cases, their acquisition results or human
judgments. If used for tuning, this set becomes development data and another
untouched confirmation set is required. This is a small structured question-level
sample, not random clinical sampling, disease/paper disjointness or expert validation.
