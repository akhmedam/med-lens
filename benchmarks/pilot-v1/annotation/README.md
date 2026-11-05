# Pilot judgments

The small label files are tracked at their original `data/evaluation/pilot-v1/`
paths so historical profiles and SHA-256 bindings remain valid. See
`manifest.json` for exact identities; `query-intents.json` preserves the original
question-specific intents. General grades and metric definitions are in
[the evaluation guide](../../../docs/evaluation.md).

- v2: 200 pairs, 198 judged; Codex title/primary-abstract assessments. The rationale
  and confidence records are in `reviews-v2.json`.
- v3: all 200 v2 pairs unchanged, plus ten supplementary assessments. It has 210
  pairs, 208 judged; supplementary rationale/confidence records and provenance
  are tracked alongside its labels.
- Files named `human-*`, `dense-supplement-human-*` and `rrf-supplement-human-*`
  preserve earlier user annotation rounds. They are historical inputs, not the
  current v3 labels.

These are development judgments, not independent expert ground truth. Two
missing abstracts remain null; absent pairs remain unjudged. The original age
policy differs between question intents; it was not retrospectively harmonized.
Publishing labels does not make this a held-out benchmark. The separate test
questions are reserved for Day 26.

Primary abstracts, source snapshots, annotation pages and raw experiment reports
remain local. The transfer bundle restores those exact evidence files; cloning
alone supplies labels and aggregate reports, not the frozen corpus.
