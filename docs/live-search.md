# Live PubMed acquisition

The [supported MVP](mvp.md) performs a fresh bounded PubMed title/abstract search
for each English biomedical research question. The primary command is `medlens`;
no prepared local collection is required. The local API uses the same
acquisition and source-findings policy.

```sh
.venv/bin/medlens --question 'How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?' --limit 100
```

Use [README installation](../README.md#installation) for the main BGE environment and isolated
native Qwen4B runtime. Neither original 9B weights nor a historical source bundle
is required. The default is 100 fresh candidates and ten Dense-ranked articles;
request bounds remain 1–300 and 1–10. These are bounded runtime settings, not a
claim of exhaustive retrieval or an optimal relevance cutoff. `--search-only`
returns articles and fragments without Qwen generation. PMC full text and larger
prepared collections remain separate research work.

## Question and PubMed query

The original question serves BGE ranking and Qwen source selection. A separate
deterministic lexical planner prepares PubMed candidate-search wording:

- It removes only recorded generic English question/predicate scaffolding,
  reference-role scaffolding and conservative terminal outcome-follow-up wording.
- Quoted terms, acronym-like tokens, identifier punctuation, negation and
  ambiguous timing relations remain protected. The original question is never
  replaced with the shortened candidate expression for ranking or findings.
- No biomedical concept/disease dictionary, topic allowlist, LLM query compiler,
  language classifier or automatic translation is used. English biomedical
  questions are the supported input scope; accepting text is not language/topic
  validation.

`--pubmed-query 'EXPRESSION'` submits the exact expression unchanged. No detected
condition, topic restriction or additional filter is appended. The public JSON
records the submitted expression and transformations, PubMed's query translation,
reported total matches, actual fetched count, candidate cap and timestamps.
Nonempty candidates do not establish relevance or search completeness.

Bad input, upstream failure and incomplete downloads are explicit failures;
no older collection is silently substituted. The PubMed client is sequential at
2.5 requests/second, with a 20-second socket timeout and at most three attempts.
One CLI/API service reuses its lazy client across sequential jobs. Its limiter
is per client rather than a global coordinator across separate processes.

## Freshness and local artifacts

The CLI stores jobs under `results/cli-requests`; the API uses
`results/api-requests`. Each accepted request owns a new ID directory:

```text
REQUEST_ID/
  job.json           # public job envelope
  run/
    run.json, report.html
    snapshot/        # raw ESearch/EFetch responses and manifest
    dataset/         # parsed current records, versions and provenance
    windows/         # title/section windows and exact offsets
    embeddings/      # vectors mapped to current windows
```

PubMed is contacted even when all passage vectors are cached. Cache keys bind
exact unpadded token inputs to the pinned encoder/tokenizer policy. New metadata
can produce new mappings while unchanged inputs reuse vectors. Changed inputs
are encoded; corrupt/conflicting cache entries fail closed. Output and cache
roots must be disjoint. Source texts, vectors, weights and raw outputs stay local.

Public HTML/JSON export is explicit in the CLI. It does not copy private
`run/report.html`, prompts, diagnostic exceptions or runtime paths. The saved
source artifacts can feed the [offline research/replay wrapper](local-requests.md)
when a fixed-source experiment is needed.

## Ranking and findings

BGE-base-en-v1.5 uses normalized float32 CLS vectors, eager attention and batch
size 16 on MPS. Articles are ranked by maximum window cosine similarity. A fully
cached passage phase skips its model load; the query still needs encoding. BGE
is released before Qwen. No additional reranker is part of the application.

The pinned native Qwen3.5-4B runtime selects study-context and finding quotation
IDs. Exact source text, applicability cautions and original question scope remain
visible. There is no free factual synthesis or same-model factual-review stage
in this answer format. Nonempty outputs remain `partial`; identity checks do not
prove semantic support or completeness. Empty candidates/evidence abstain
explicitly without a Qwen call. Model failures retain completed source stages.

## Research wrapper and historical evidence

`scripts/ask_live.py` remains an explicit research wrapper with a fresh output
directory and experimental model/backend choices. To reproduce the separate
prose draft/review route, request it explicitly:

```sh
.venv/bin/python scripts/ask_live.py \
  --query 'How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?' \
  --answer-style draft --limit 100 --output-dir results/requests/research-draft-NEW
```

That route uses the current 4B compact v8/axes-first-v2 draft policy unless an
explicit historical model comparison is requested. It is not an application
answer-format switch. `--answer-model 9b`, Transformers and CPU BGE are research
options outside MVP support. Frozen measurements below keep their original
models, inputs and protocols; they are not updated with current defaults.

## Historical runtime measurements and verification

The following benchmark predates the 4B default and describes the original 9B
Transformers/Metal workflow. The live benchmark uses p03 (exercise/depression) and p06 (CBT/generalized anxiety)
from the fixed development question file. It measures 30, 100 and 300 candidates
with a fresh vector cache followed by its reuse. Both requests still search and
fetch PubMed anew, and every model call is fresh. Each request runs in a new
process; neither model nor OS storage caches are flushed. There is one observation
per question/size/cache combination, not a latency distribution or a causal
estimate of cache benefit for Qwen.

```sh
.venv/bin/python scripts/benchmark_live_search.py \
  --answer-model 9b --engine transformers \
  --output results/live/new-paired-run \
  --summary docs/experiments/live-new-paired-run.json
```

The recorded [experiment](experiments/live-search.json) binds source/code hashes,
fresh snapshot IDs, stage timings, pipeline statuses and source-integrity checks.
It does not use qrels, judge relevance, assess claim correctness or touch held-out
questions. Stage timings are nested. CLI timers exclude Python startup/imports
and the final checkpoint; command wall time includes them. Process RSS, sampled
MPS allocations and Qwen conversion counters overlap and must not be added.

The added integration tests exercise fresh refetching with vector reuse, source
membership changes, changed text, empty results without model loads, corrupt
caches, preserved evidence after generation failure, upstream failure without
stale fallback, output ownership and query planning. They use fictional sources
and deterministic/fake model outputs, not clinical labels. The end-to-end runs
check real saved source spans and visible citations, independently of any verdict
about what a generated claim means.

## Historical 9B observations

| Question | Candidates | Cold command | Warm command | Cold passage prep | Warm passage prep | Automatic answer status (cold / warm) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| p03 | 30 | 142.33 s | 163.32 s | 4.68 s | 0.90 s | partial / partial |
| p03 | 100 | 205.57 s | 188.04 s | 14.35 s | 1.72 s | validation_failed / validation_failed |
| p03 | 300 | 224.37 s | 188.11 s | 47.17 s | 1.09 s | answered / answered |
| p06 | 30 | 196.06 s | 191.18 s | 5.47 s | 0.03 s | answered / answered |
| p06 | 100 | 214.84 s | 202.84 s | 18.16 s | 1.04 s | answered / answered |
| p06 | 300 | 292.21 s | 205.94 s | 73.29 s | 1.32 s | insufficient_evidence / insufficient_evidence |

Fresh download took 1.93–5.37 s in these runs. The largest process RSS high-water was 4.73 GiB; the largest separately sampled Qwen conversion driver allocation was 9.04 GiB. These are overlapping counters, not values to add. No system-wide or swap peak was measured.

Automatic statuses: partial: 2, validation_failed: 2, answered: 6, insufficient_evidence: 2. These are not correctness labels. Validation failures kept search and source evidence while withholding the answer; no enum repair or prompt change was applied.

Warm calls fetched PubMed anew and had changing candidate memberships; they are not a controlled identical-source speed comparison. The initial cap of 100 bounds measured preparation cost while inspecting more candidates than the smallest setting, without establishing a relevance optimum. The 300-candidate cold run required up to 73.29 s for passage preparation alone. Qwen remains the dominant stage in the full requests.

The [verification record](experiments/live-search-checks.json) covers 288 passing unit tests, dependencies, twelve distinct live snapshots, source spans and visible citations. A real empty-result query abstained without loading weights; synthetic upstream failures did not fall back to old data. The user's manual annotations were not replaced or extended.
