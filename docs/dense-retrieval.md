# Exact dense retrieval

MedLens can now rank a frozen PubMed candidate set with BGE vectors. Search uses
an exact NumPy dot product of unit vectors, followed by article aggregation. No
ANN service is needed for the current 30-candidate pools.

```sh
python -m pip install -e '.[retrieval,embeddings]'
python scripts/dense_search.py data/parsed/pilot-v1/p10 \
  --windows data/windows/pilot-v1-002/p10 \
  --embeddings data/embeddings/pilot-v1/p10 \
  --query 'Does cognitive behavioral therapy reduce anxiety symptoms in children with anxiety disorders?' \
  --output results/dense/p10-001.json
```

The default model/tokenizer directories are the local caches created in Day 8.
Choose `--model`, `--tokenizer`, or `--device cpu` explicitly to override them.
Inputs are validated before query encoding; no network calls or automatic model
downloads occur. Existing reports are refused. Queries use the same pinned BGE
model, CLS pooling, normalization, and retrieval instruction as the stored vectors.
CPU and batch-size changes are allowed, but encoder policy and tokenizer metadata
must match the passage artifacts.

## Scoring and provenance

Window similarity is `dot(unit_window, unit_query)`. Cosine distance is
`1 - similarity`, with higher similarity / lower distance ranked first. Values
are clipped to `[-1, 1]` only to handle floating-point roundoff. Article score is
the **maximum** over its windows. A paper appears once regardless of window count.
Equal article scores retain the candidate order; equal windows retain the first
source window. Zero and negative similarities remain eligible. No score threshold
has been tuned on the pilot.

Dataset ID, article-file hash, window identity, ordered vector mapping, and text
versions must agree. Articles without vectors are listed explicitly and excluded
from scoring, rather than assigned a fabricated zero. Every result contains the
winning window ID, source span, exact excerpt, provenance, and PubMed URL. This
explains which fragment matched; similarity alone does not establish relevance
or support a generated claim.

Run reports record input artifact IDs, original query, query-vector hash, encoder
settings/code hashes, score/tie policy, and separate query-encoding/search timings.
The run ID depends on the actual query-vector bytes; floating-point differences
across runtimes may change it even when rankings remain the same.

## Pilot comparison

```sh
python scripts/evaluate_dense.py benchmarks/pilot-v1/queries.jsonl \
  --qrels data/evaluation/pilot-v1/human-final-001.jsonl \
  --windows-root data/windows/pilot-v1-002 \
  --embeddings-root data/embeddings/pilot-v1 \
  --output results/evaluation/pilot-v1-dense-001.json
```

The evaluator preserves PubMed and BM25 baselines on the same candidates. The
saved run covered all 300 candidate memberships / 975 windows. All candidates
had vectors. Ten exact searches took 2.36 ms total, excluding artifact validation,
model loading, and query encoding. The ten separately encoded queries took
0.44 seconds on MPS in that run. These are observations, not end-to-end benchmarks.

Only **62 of the 100 dense top-10 positions** have prior human labels. The
remaining **38 query/PMID pairs** are new to the annotation pool. The existing
161 judgments fully cover PubMed/BM25 top-10 but do not permit a complete dense
comparison. The [aggregate report](experiments/dense-retrieval.json) is explicitly
marked provisional. Its numeric pooled scores treat unjudged as zero contribution;
they must not be interpreted as proof that dense retrieval is better or worse.
They are not guaranteed lower bounds either: additional positive labels also
change ideal DCG and the known-relevant denominator for all methods.

## Additional annotation

The supplemental packet contains only the missing 38 cards, in shuffled order,
without source ranks or method scores. Previous judgments remain unchanged.

```sh
python scripts/annotate_pilot.py build benchmarks/pilot-v1/queries.jsonl \
  --plan benchmarks/pilot-v1/plan.json \
  --dense-report results/evaluation/pilot-v1-dense-001.json \
  --qrels data/evaluation/pilot-v1/human-final-001.jsonl \
  --output data/annotation/pilot-dense-supplement-001
python -m http.server 8766 --bind 127.0.0.1 --directory data/annotation
```

Open `http://127.0.0.1:8766/pilot-dense-supplement-001/`. Download its annotations
and validate/merge them into a **new** qrels file:

```sh
python scripts/annotate_pilot.py export data/annotation/pilot-dense-supplement-001 \
  --annotations /path/to/downloaded-annotations.json \
  --output data/evaluation/pilot-v1/dense-supplement-001.jsonl
python scripts/annotate_pilot.py merge benchmarks/pilot-v1/queries.jsonl \
  --base data/evaluation/pilot-v1/human-final-001.jsonl \
  --supplement data/evaluation/pilot-v1/dense-supplement-001.jsonl \
  --pool data/annotation/pilot-dense-supplement-001 \
  --output data/evaluation/pilot-v1/human-expanded-001.jsonl
```

The supplement binds exact query, dataset, report, and base-qrels identities.
Merge requires a complete validated supplement, rejects changed inputs and
conflicting labels, preserves prior grades, and writes a provenance receipt.
Rerun `evaluate_dense.py` with the expanded qrels and a new output path. The
resulting 199 judged pairs will be development evidence, not a held-out test.
Future RRF retrieval may require another pool expansion.

All 101 offline tests pass. Cases include an independent cosine reference,
max-versus-mean scoring, deterministic ties, negative scores, invalid vectors,
artifact mismatches, CLI output, unjudged coverage, supplemental packet selection,
and preserving old labels during merge. A real-model CLI check and ten-query
evaluation also completed. No new relevance labels were generated by the code.
