# Shared local corpus and embedding cache

Ten rebuilt development collections now form one searchable local collection:
**300 article memberships → 277 unique PMIDs, 850 source windows and 846 unique
BGE inputs**. This starts the Small → Medium transition on the Air M3 (24 GiB).
It does not establish large-corpus capacity or improved retrieval/answer quality.
The original historical snapshots and raw experiments remain lost.

## Build and use

With the pinned assets and the recovery profile available, choose a fresh revision:

```sh
.venv/bin/python scripts/pool_collections.py build \
  --collections configs/collections/recovery-20261106.json \
  --output data/collections/shared-example
.venv/bin/python scripts/pool_collections.py inspect data/collections/shared-example
.venv/bin/python scripts/ask.py data/collections/shared-example/dataset \
  --windows data/collections/shared-example/windows \
  --embeddings data/collections/shared-example/embeddings \
  --query "Does cognitive behavioral therapy reduce depressive symptoms in adults with depression?" \
  --generate --output-dir results/requests/shared-example
```

The recorded benchmark's usable revision is
`data/collections/shared-20261108/shared`. The ordinary primary CLI accepts its
three artifact directories directly. Generation stays at draft v6 / baseline
review v4 with the existing Qwen revision, FP16 Metal 4-bit/group-64 runtime and
input/output budgets. Do not add `--rerank` for the accepted primary workflow.
Pooled loading currently supports Dense and the primary request pipeline;
historical BM25/MedCPT runners still expect native topic datasets.

Use repeatable `--source p01` options to build a subset. Source labels and resource
identities come from the verified collection profile. A build reads all selected
native artifacts, reconstructs their windows with the actual pinned tokenizer,
and verifies profile identities before importing any vectors. It makes no NCBI
requests, changes no input artifacts and reads no relevance labels.

## Records, versions and provenance

A revision retains copied **parsed native observations** under
`dataset/sources/<dataset_id>/`, a catalog of every observation/text version and
one active article per PMID. The selected article dictionary is preserved exactly,
including metadata, provenance, source paths and offsets. The copied observation
files are sufficient for local verification after the input folders are removed;
this is not an additional backup of raw XML snapshots or weights.

The newest timezone-aware *retrieval* timestamp chooses the active observation.
It does not select by publication date or evidence quality. Materially different
records tied at the newest timestamp fail closed; provide
`--resolutions <json-file>` with `{ "<pmid>": "<latest-native-dataset-id>" }`.
Equivalent tied records differing only in recursive provenance/source paths select
by dataset ID deterministically. An absent or older resolution is rejected.
Every text-version identity is recomputed using the existing parser policy.
Active articles are ordered by numeric PMID; this also defines Dense article ties.

Windows and vector mappings are rebuilt for the active records. A metadata-only
update can change citation/window identities while reusing vectors if input IDs
stay identical. A changed text/window policy encodes only missing inputs. Removal
rebuilds active memberships and mappings without deleting old immutable revisions
or cache entries. The current real observations have **no normalized text-version
changes**: 20 repeated PMIDs account for 23 excess memberships. Changed text,
metadata updates, conflicting ties and failed updates are covered by synthetic
tests, rather than claimed as observed historical changes.

## Cache lifecycle and publication

The default cache is `.cache/embeddings/bge-input-v1`. Keys bind exact unpadded
input IDs, pinned encoder/tokenizer identities, package/device/batch policy and
encoding/cache implementation hashes. Changes to those policies cause misses.
Titles, IDs and provenance remain in windows and mappings rather than being
substituted into a cached vector. Equal inputs may share a vector across PMIDs.

Each cache entry has a checksummed JSON header and exactly 768 float32 values.
Finite unit norms, input/policy/vector identities and file size are verified;
symlinks are rejected. Concurrent writers use exclusive hard-link publication.
Existing entries are never replaced: differences up to `1e-5` retain the earlier
vector and record the drift; larger conflicts or corrupt entries stop the build.
Do not delete one corrupt file to silently make a run succeed. Investigate it
and select an explicitly fresh cache root when a clean rebuild is needed.

Native source vectors seed the cache only when their verified implementation,
encoder and tokenizer policies match. `--no-seed` suppresses native imports;
`--reuse <previous-shared-revision>` can seed a new empty cache from a verified
shared index. Old entries survive membership removal. There is no automatic
cache eviction; choose a fresh cache namespace for a clean lifecycle experiment.

Output, cache, reuse and input artifact roots must be disjoint. Outputs must be
fresh. A running revision becomes readable only after validation and atomic
publication of its complete root manifest; failures receive a failed marker.
Readers reject failed/incomplete roots. This is **not an atomic directory swap**,
and there is no implicit active pointer. Keep using the previous complete
revision if an update fails. Logical collection identity excludes measurements;
the root manifest identity also binds timings, creation time and payload checksums.

## Measured preparation and search

The [aggregate record](experiments/shared-corpus.json) binds code, input files,
artifact IDs and runtime versions. No source bodies or raw model output are
tracked. Reproduce in fresh paths:

```sh
.venv/bin/python scripts/benchmark_shared_corpus.py \
  --collections configs/collections/recovery-20261106.json \
  --output data/collections/shared-new-benchmark \
  --repeats 5 --summary docs/experiments/shared-new-benchmark.json
```

| Preparation scenario | Wall time | Encoded inputs / model loads |
| --- | ---: | ---: |
| Small p01, seed native vectors | 0.123 s | 0 / 0 |
| Grow to all ten collections, seed native vectors | 1.371 s | 0 / 0 |
| Repeat full revision from the cache | 0.989 s | 0 / 0 |
| Reverse input order, cached | 0.975 s | 0 / 0 |
| Remove back to p01, cached | 0.090 s | 0 / 0 |
| Seed an empty cache from the previous shared revision | 1.802 s | 0 / 0 |
| Empty vector cache, actual BGE MPS encoding | 33.086 s | 846 / 1 |

These are single runs with filesystem caches left intact, from verified native
parsed/window/vector artifacts through final shared validation. They exclude
Python startup, standalone tokenizer loading, downloads, parsing from XML and
window preparation of the original native inputs. "Empty" describes the vector
cache, not OS/model storage caches. The fresh run includes 0.630 s encoder
construction, 30.892 s encoding and 0.479 s cache publication. Reordered/repeated
builds and reseeding preserve logical identities; removal restores the Small
identity. Every input file hash remained unchanged.

Search used the ten fixed **development** queries, five repetitions each, one
resident encoder, validated in-memory indexes and excluded warm-up. Each query
was encoded once per repetition and the same vector served both searches;
Small/shared search order alternated. Small here means each query's original
30-article topic collection, not only p01.

| Stage | Median |
| --- | ---: |
| BGE query encoding, 50 calls | 12.355 ms |
| Exact Small search/aggregation, 50 calls | 0.091 ms |
| Exact shared search/aggregation, 50 calls | 0.403 ms |
| Small index validation, ten indexes | 5.409 ms |
| Shared index validation, one index | 169.008 ms |

Search-worker encoder construction was 2.157 s, separate from resident query
times. The pooled matrix uses 2,611,200 bytes. Cold/cached vectors differed by at
most `1.59e-7`; all fifty shared top-10 rankings matched. Small/shared top-10
overlap ranged from 6 to 10 articles. Overlap is descriptive, not a relevance
metric. No historical qrels, nDCG/recall or held-out data entered this experiment.

Fresh-build process peak RSS was 0.780 GiB; sampled post-forward MPS tensors were
0.428 GiB and MPS driver allocation 2.008 GiB. Search-worker peak RSS was
0.773 GiB. These counters overlap and must not be added; sampled device counters
are not instantaneous device peaks. Consult the record for all scopes and raw
byte counts. Retrieval costs at 277 articles are not a scaling forecast.

## Checks and remaining limits

The shared reader checks copied-source lineage, active selection, text versions,
ordered vector mappings, unit vectors and source spans. The full suite contains
276 passing tests, including 16 new cache/corpus cases. The accompanying
[request smoke record](experiments/shared-corpus-request.json) checks the ordinary
primary CLI against the shared revision, with unchanged strict generation gates.
The request completed in 136.777 s (excluding Python startup/imports), loaded
Qwen once and retained three draft claims. All six selected excerpts, five visible
citations and 28 quote-catalog entries matched their sources; replay of both
saved model calls reproduced the result without inference. This single smoke
does not independently assess the claims or establish a success rate.

The [semantic error record](day21-semantic-reliability.md) still applies. A larger
candidate pool may change retrieved evidence and generated drafts; neither valid
format nor exact quotations establish support. Independent relevance judgments
and answer assessment and a larger measured Medium corpus remain future work.
The [PMC study](pmc-fulltext.md) now measures a six-article licensed subset; full
text stays opt-in until its relevance and answer value are independently assessed.
Publication recency is not a quality signal.
