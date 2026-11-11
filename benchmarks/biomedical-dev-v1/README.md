# Biomedical development annotation plan

This is a six-question English biomedical **development** set, frozen on
2026-11-11 before source acquisition and independent human annotation. It does
not use the historical qrels or existing held-out questions. No relevance or
answer-quality metrics are permitted before the corresponding human reference
has been completed and explicitly finalized.

Preparation completed on 2026-11-11: six search-only source runs froze 246
candidate records across cases and a verified **84-card blank relevance package**.
The local form is
`results/biomedical-dev-v1/manual-annotation-20261111-v1/index.html`; its package ID
is `8603f85b744d83f54f9fc3fbd42a6ad8aae2c8f8d88bf7d6539417651ca955a5`.
The [public integrity record](../../docs/experiments/biomedical-scope-preparation.json)
records the bindings and functional checks. All human decisions are still pending;
no answers were generated for these source cases and no quality scores were computed.
Automated checks passed within their recorded scope. Visual browser QA was not
performed because the browser tool rejected local-file navigation; the real form
remains blank and untouched.

The authoritative [plan.json](plan.json) defines the exact questions, relevance
intents, article rubric and separate answer-evaluation protocol. Its byte SHA256
is `eaece352cff4dc483d242dab22c9b32eba657a28e36be9f9c8dc41ec95ed497e`, recorded in
[plan.sha256](plan.sha256). Do not edit either file to relabel an existing round.
A changed question or protocol requires an explicitly new plan/version.

| ID | Specialty | Question type | Fetched records | Annotation cards |
| --- | --- | --- | ---: | ---: |
| bm01 | Cardiology | Prevention | 66 | 17 |
| bm02 | Endocrinology | Treatment comparison | 13 | 12 |
| bm03 | Infectious diseases | Diagnosis | 44 | 17 |
| bm04 | Neurology | Prognosis | 100 | 20 |
| bm05 | Immunology and metabolism | Mechanism | 19 | 14 |
| bm06 | Pediatrics | Treatment | 4 | 4 |

Each case freezes up to 100 current PubMed candidates. Its blind annotation pool
is the union of up to ten PubMed Best Match articles and up to ten Dense articles
from that exact snapshot, deduplicated by PMID. Rankings, scores and method names are omitted
from the annotation view. All cards start pending. Only the project owner supplies
0/1/2 relevance decisions or explicit unjudgeable decisions and any rationale.
There are no prefilled labels, suggested grades, model answers or assistant-written
annotation rationales. Preserve unknowns and original exported bytes.

Use the [preparation, annotation and freeze workflow](../../docs/newbroader-biomedical.md)
for the exact CLI commands. Source snapshots, article text, vectors and browser
forms remain local; this directory tracks the frozen preparation protocol.
Successful acquisition or format/citation checks are functional evidence only.

Article relevance and answer correctness require distinct human references.
Finalize and freeze relevance judgments before retrieval scoring. Generated
answers later need separately frozen human claim/answer judgments before any
answer-quality score. Missing or unjudgeable judgments never silently become
negative labels. Once these cases guide tuning, they are not independent test data.
