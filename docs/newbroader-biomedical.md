# Broader biomedical development and human annotation

The development workflow now accepts English biomedical questions beyond the
historical depression/anxiety cases. This is a functional scope change. Retrieval
relevance and answer quality on the new topics remain unmeasured until the user
independently completes and explicitly finalizes the required human annotations.
The [scope and human-reference decision](adr/0006-biomedical-scope-and-human-reference.md)
records these separate gates.

The source is PubMed titles and abstracts, with a bounded candidate set per
request; this is not unrestricted web search or a multilingual evaluation. The
primary chain remains fresh PubMed acquisition → cached/new BGE passage vectors
→ Dense article ranking → exact evidence → native MLX Qwen3.5-4B draft and
same-model review. PMC full text remains a separate opt-in experiment.

## Prepared package and current status

Source and blank-package preparation completed on 2026-11-11. The verified local
package is `results/biomedical-dev-v1/manual-annotation-20261111-v1`, with package ID
`8603f85b744d83f54f9fc3fbd42a6ad8aae2c8f8d88bf7d6539417651ca955a5`.
It contains six questions and **84 question/article cards**, all still blank and
awaiting the user's independent decisions. The six source requests were search-only:
no Qwen weights were loaded and no answers or human relevance labels were generated
for these cases. The [preparation and integrity record](experiments/biomedical-scope-preparation.json)
contains source, package and implementation identities.

| Case | Fetched candidate records | Pooled annotation cards |
| --- | ---: | ---: |
| bm01 | 66 | 17 |
| bm02 | 13 | 12 |
| bm03 | 44 | 17 |
| bm04 | 100 | 20 |
| bm05 | 19 | 14 |
| bm06 | 4 | 4 |
| Total across cases | 246 | 84 |

These are preparation counts, not relevance or recall estimates. The source runs
are under `results/biomedical-dev-v1/sources-20261111-v2`. Open the existing local
blank form to begin annotation; the later reproduction commands are unnecessary
for this prepared round:

```sh
open results/biomedical-dev-v1/manual-annotation-20261111-v1/index.html
```

Automated validation completed with 394 passing unit tests and seven optional
main-environment skips, including ten annotation tests. Eight JavaScript VM checks
used wholly fictional three-item fixtures; an exported SHA256 was cross-checked
with Python. Wheel construction, template inclusion, dependency checks and frozen
package verification passed. A separate native v8 draft/review smoke on fictional
sources completed in 25.18 s. None of these checks supplies biomedical quality labels.

**Visual browser QA remains unperformed:** the browser tool rejected local
`file://` navigation, and no alternate browser route was attempted. The JavaScript
checks verify logic rather than rendered layout. The real 84-card form remains
untouched. This is a documented QA limitation; annotation preparation is complete,
while human annotation and both quality-evaluation gates remain pending.

## Query construction and limits

The deterministic `lexical-question-scaffold-pubmed-atm-v1` planner removes generic
English question scaffolding before PubMed submission. It removes supported
question/auxiliary phrases and standalone generic predicates such as compare or
reduce; it does not use a disease/intervention dictionary or an LLM. Medical,
population, time and named-comparator terms are retained. Reference-role phrasing
can be removed while the named reference test remains. Double-quoted text, acronym-like
tokens, compound identifiers and directly negated predicates are protected.

The original question is unchanged for BGE and Qwen. The request records each
query transformation, its removed text and original character spans, whitespace
normalization, the exact submitted query and PubMed's returned query translation.
This is a transparent lexical heuristic, not a validated clinical PICO compiler.
PubMed Automatic Term Mapping can still interpret terms unexpectedly, and
natural-language wording can narrow the candidate set. Functional nonempty search
results do not establish relevance or complete topic coverage.

Use `--pubmed-query '<PubMed expression>'` when explicit search syntax is needed.
The `explicit-pubmed-query-v2` route submits that expression literally: it does
not append a detected disease or other hidden filter. The question remains the
research question for ranking and generation. Language/topic classification and
automatic translation are not performed; English biomedical research is the
supported development scope, not a claim that every input is recognized correctly.

## Question-driven evidence and review

The existing selector was audited rather than tuned on new labels. It retains
`article-round-robin-section-priority-window-cosine-no-overlap-v2`: round-robin
articles, generic results/conclusion section priority, window cosine order,
nonoverlapping source spans, at most two fragments per article and six fragments
in total. It has no depression/anxiety, adult-only or treatment-efficacy filter.
This audit is not evidence of improved relevance; the 4,000-token evidence bound
still applies.

The default compact draft is `cited-answer-v8`; its six-field v4 reviewer is
`axes-first-v2`. Treatment, prevention, diagnosis, prognosis, mechanism and other
biomedical intents follow the question. Requested population, condition, age,
species, experimental system, intervention, comparator and outcome constraints
remain binding. Unrequested adult age, treatment benefit and untreated controls
are not universal prerequisites. Human, animal and in-vitro findings preserve
their actual source scope; they do not establish unrestricted clinical benefit.

An explicitly wrong requested scope remains wrong even when accurately labeled.
Missing requested details remain uncertain. Separate A-versus-C and B-versus-C
findings do not establish A-versus-B; a question without a requested contrast does
not acquire an invented control group. Exact quotations, consistent JSON and
same-model support review do not establish scientific correctness.

The maximum three claims, one to three exact citations per claim, 4,000-token
evidence budget, 6,500-token full input limit and 768/512 draft/review token caps
remain unchanged. Historical 9B prompts remain separate. The earlier 4B v7 human
assessment and integration timings retain their original development scopes;
they are not evidence of quality or latency for the new biomedical workflow.

## Frozen development protocol

The [plan](../benchmarks/biomedical-dev-v1/plan.json) was frozen before source
acquisition and human annotation. Its byte SHA256 is
`eaece352cff4dc483d242dab22c9b32eba657a28e36be9f9c8dc41ec95ed497e`, also recorded in
[plan.sha256](../benchmarks/biomedical-dev-v1/plan.sha256). It defines six English
**development** questions:

| ID | Specialty | Intent |
| --- | --- | --- |
| bm01 | Cardiology | Prevention of heart-failure hospitalization |
| bm02 | Endocrinology | Comparative diabetes treatment |
| bm03 | Infectious diseases | Diagnostic accuracy |
| bm04 | Neurology | Prognosis/progression association |
| bm05 | Immunology and metabolism | Biological mechanism |
| bm06 | Pediatrics | Treatment and a specified follow-up interval |

The exact questions and relevance intents live in the frozen plan. These intents
specify what the user judges, not a suggested grade or an expected answer. Neither
the historical qrels nor the existing held-out questions are used.

Each question acquires a fresh snapshot of up to 100 PubMed Best Match candidates.
The annotation pool is the union of up to ten first PubMed candidates and up to ten
Dense articles **from that same snapshot**, with duplicate PMIDs merged. Each question
therefore has at most 20 article cards. This pool supports a bounded method
comparison, not recall over every matching PubMed article. Its size, overlap and
source identities are structural facts, not relevance metrics.

The source-bound package preserves exact title/abstract versions and their
hashes, snapshot identity, question/PMID bindings and frozen implementation files.
The annotation page hides retrieval method, score and rank and uses deterministic
blind ordering. Provenance remains available in the audit files. No model answers,
relevance grades or model-written rationales are supplied.

## Prepare a new source and annotation package

These commands create new local artifacts and refuse an existing output path.
They are for reproducing preparation, not for replacing an existing annotation
round after judging has begun. The source command uses search-only requests and
loads no Qwen weights:

```sh
.venv/bin/python scripts/prepare_biomedical_cases.py \
  --plan benchmarks/biomedical-dev-v1/plan.json \
  --output results/biomedical-dev-v1/new-sources

.venv/bin/python scripts/prepare_biomedical_annotation.py build \
  --plan benchmarks/biomedical-dev-v1/plan.json \
  --runs-root results/biomedical-dev-v1/new-sources \
  --output results/biomedical-dev-v1/new-annotation

.venv/bin/python scripts/prepare_biomedical_annotation.py verify \
  --package results/biomedical-dev-v1/new-annotation
open results/biomedical-dev-v1/new-annotation/index.html
```

The package contains `package.json`, `manifest.json`, `items.json`, a frozen
`plan.json`, `index.html`, source artifacts under `sources/` and implementation
copies under `implementation/`. Verification checks these bindings against the
frozen sources, rather than accepting edited presentation text as a new reference.
Snapshots, article text, vectors and annotation pages remain local artifacts.

## The user annotates independently

Every card starts **pending**, with no default relevance value. The user chooses:

| Decision | Meaning |
| --- | --- |
| 2 | Directly relevant to the main target and all explicitly requested restrictions; a null result can be directly relevant. |
| 1 | Useful partial evidence, without directly answering the complete requested target or restrictions. |
| 0 | No useful evidence for the requested question or a materially different target. |
| Unjudgeable | The available title/abstract cannot be interpreted confidently or is insufficient to decide. |

These are article-relevance decisions, not study-validity or answer-correctness
labels. Unjudgeable is an explicit human decision; it is different from an
untouched pending card and from relevance zero. The user alone enters any rationale.
The assistant neither fills labels nor suggests decisions or annotation rationales.

Progress is saved in the current browser. Export a draft before changing browser
or computer; importing the user's own JSON restores its unchanged decisions.
Finalize only after every pooled card has an
explicit decision: pending must be zero, while explicit unjudgeable decisions are
allowed and remain visible. Keep the exported original JSON unchanged. Use its
actual path in the validation and freeze commands:

```sh
.venv/bin/python scripts/prepare_biomedical_annotation.py validate \
  --package results/biomedical-dev-v1/new-annotation \
  --human-export /path/to/the-export.json --require-finalized

.venv/bin/python scripts/prepare_biomedical_annotation.py freeze \
  --package results/biomedical-dev-v1/new-annotation \
  --human-export /path/to/the-export.json \
  --output results/biomedical-dev-v1/finalized-human-reference
```

Validation checks structure, checksums, finalization and exact item/source
bindings; it does not assess whether a human grade is scientifically correct.
Freeze preserves the original export bytes as read-only
`original-human-export.json` and writes `receipt.json` with provenance and
coverage counts. It produces no quality score and does not change human values.
A validation error requires investigation or a corrected export by the user,
not an assistant-authored semantic correction.

## Quality evaluation follows its own human references

Only a completed and explicitly finalized article reference can support later
retrieval evaluation. Unknown decisions remain separate, judgment coverage must
be reported, and any recall denominator must be identified as pool-limited.
No provisional, absent or unjudgeable label silently becomes irrelevant.

Answer quality is a separate stage. First freeze the outputs and a separate blank
human claim/answer form; the user then independently judges factual support,
requested population/system, comparison, outcome/target and preserved qualifiers,
plus answer support, scope, completeness and usability. Freeze that explicitly
finalized reference before computing answer-quality metrics. Article relevance
alone cannot establish any of those answer properties.

The new set is for development. Later tuning on it prevents presenting it as an
independent held-out test. Runtime, integrity and fictional contract tests may
proceed without human labels; none produces a broader-topic quality estimate.
