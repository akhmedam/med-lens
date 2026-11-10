# Live PubMed search-to-answer requests

`scripts/ask_live.py` performs a fresh PubMed search and fetches current abstract
records for each request, then runs the selected Dense → evidence → Qwen draft
and support-review pipeline. It does not require a prebuilt local collection.
The current scope remains English questions about depression, anxiety and panic;
broader biomedical evaluation follows the user's independent manual annotation.
Use the pinned local assets prepared by the [recovery guide](recovery-without-bundle.md);
restoring an old source collection is not required for this mode.

```sh
.venv/bin/python scripts/ask_live.py \
  --query "Does exercise reduce depressive symptoms in adults with depression?" \
  --limit 100 --output-dir results/requests/live-exercise-example
open results/requests/live-exercise-example/report.html
```

Generation is enabled by default. `--search-only` runs diagnostic retrieval and
source selection without Qwen inference. `--top-k` accepts 1–10 articles. The live
candidate cap is 1–300; the initial default of 100 is a bounded runtime choice,
not a relevance optimum. It needs reassessment after the user's new labels exist.
The existing `ask.py` continues to accept verified local collections for replay
and experiments. PMC full text remains a separate opt-in experiment.

## Question and PubMed query

The original question serves BGE and Qwen. A separate, deterministic query planner
extracts supported condition and intervention concepts for PubMed Best Match:

- Conditions: depression, generalized anxiety/GAD, social anxiety and panic, with
  a generic anxiety fallback.
- Intervention terms: CBT, antidepressants/SSRIs, exercise, mindfulness/MBCT and
  exposure. Detected interventions form an OR group so comparison questions can
  retrieve evidence mentioning either intervention.
- Age, severity, outcomes and delivery qualifiers remain in the original question.
  They are not automatically added as retrieval filters. This is broad candidate
  collection followed by semantic ranking and source review, not a comprehensive
  clinical query compiler or a validated topic classifier.

Use `--pubmed-query '<PubMed expression>'` for concepts the fixed vocabulary does
not capture. The submitted expression is combined with the condition detected in
the question and is recorded verbatim. The report displays the submitted query,
PubMed's translation, total reported matches, fetched count, limit and timestamps.
There is no claim to fetch every matching paper. Bad syntax, upstream failures or
incomplete responses fail the request with its saved source checkpoint. The
existing client sends requests sequentially at 2.5/s, with a 20 s socket timeout
and at most three attempts. Its limiter is per client, not an IP-wide coordinator;
concurrent request orchestration belongs to the later API lifecycle work.

## Freshness and local artifacts

Each fresh request owns a new directory:

```text
request/
  run.json, report.html
  snapshot/     # raw current ESearch and EFetch responses and manifest
  dataset/      # verified current parsed records with metadata/provenance
  windows/      # native title + section windows and exact offsets
  embeddings/   # current vector rows mapped to current windows
```

PubMed is contacted again even when all passage vectors exist. The cache stores
only vectors keyed by exact unpadded token inputs and the pinned encoder/tokenizer
policy; new metadata/provenance create new window mappings without unnecessary
inference. Changed inputs cause new encoding. Corrupt/conflicting cache entries
fail closed, and request failure never silently substitutes an older collection.
Output and cache roots must be disjoint. Existing request directories are refused.
Snapshots, source text, vectors and raw model outputs stay ignored by Git.
The saved dataset/windows/embeddings directories can be passed to the existing
offline `ask.py` to keep evidence fixed for later manual annotation and evaluation.

The default passage cache is `.cache/embeddings/bge-input-v1`. BGE remains pinned
to `BAAI/bge-base-en-v1.5` with normalized float32 CLS vectors, eager attention and
batch size 16 on MPS. On a cold vector cache, passage encoding and query encoding
currently use separate sequential BGE instances. A request whose passage inputs are all cached skips the passage
model entirely; fresh result changes can still require some new vectors. BGE is collected and unused MPS cache is released before loading
Qwen. One Qwen instance generates and reviews the answer with the current v6/v4
contracts and token budgets. The default is now the [isolated native MLX
backend](native-mlx.md); `--engine transformers` selects the original runtime.
Both record their exact precision and stop policy. The measurements below
remain the original Transformers/Metal baseline; see the native runtime record
for the separate optimization experiment.

Empty PubMed results complete with an explicit insufficient-evidence result and
load no model weights. If all selected sources lack usable excerpts, the ordinary
context stage can also abstain without Qwen. Upstream/model errors retain completed
source/retrieval stages and identify the failed stage. Passing format/citation or
same-model review gates does not establish semantic correctness.

Exit codes follow the existing request CLI: 0 for a completed request including
abstention, 1 for execution/source errors, 2 for a completed validation failure or
argument error (inspect `run.json`), and 130 for interruption.

## Runtime measurements and verification

The live benchmark uses p03 (exercise/depression) and p06 (CBT/generalized anxiety)
from the fixed development question file. It measures 30, 100 and 300 candidates
with a fresh vector cache followed by its reuse. Both requests still search and
fetch PubMed anew, and every model call is fresh. Each request runs in a new
process; neither model nor OS storage caches are flushed. There is one observation
per question/size/cache combination, not a latency distribution or a causal
estimate of cache benefit for Qwen.

```sh
.venv/bin/python scripts/benchmark_live_search.py \
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

## Recorded observations

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
