# BM25 + dense with RRF

Reciprocal rank fusion is now available as an explicit retrieval experiment.
It combines article ranks, not BM25 scores and cosine similarities. The initial
configuration uses equal weights, `c=60`, source depth 30, and top-10 output.
These settings were fixed before examining the pilot result, not tuned on its
ten questions.

```sh
python scripts/hybrid_search.py data/parsed/pilot-v1/p08 \
  --windows data/windows/pilot-v1-002/p08 \
  --embeddings data/embeddings/pilot-v1/p08 \
  --query 'Does exposure-based cognitive behavioral therapy reduce panic symptoms in adults with panic disorder?' \
  --output results/hybrid/p08-001.json
```

Use `--c`, `--depth`, and `--top-k` to specify another experiment. Smoothing must
be finite and nonnegative; `0 < top-k <= depth`. Model/tokenizer paths and device
flags match the dense CLI. Everything runs offline on saved candidates. Existing
output files are refused. BM25 uses its existing default configuration; dense
uses the verified BGE artifacts and maximum-window article scoring. The query
is embedded once per hybrid run.

## Fusion policy

For an article d:

```text
RRF(d) = contribution_bm25(d) + contribution_dense(d)
contribution_m(d) = 1 / (c + rank_m(d)) if present, otherwise 0
```

Ranks start at **1**. With `c=60`, the first contribution is `1/61`, not `1/60`.
Each method contributes at most once per PMID. An article absent from both lists
is not returned. Empty/short lists are valid; duplicate or unknown PMIDs and
inconsistent source identities are rejected. Tied RRF scores retain the original
candidate order. Changing either method's score scale cannot affect fusion if
its rank order stays the same.

Only the first `depth` results from each method contribute. BM25 still omits
zero-score articles; dense still retains negative similarities. Depth therefore
affects both membership and ranks and is included in the run identity. On the
30-candidate pilot, depth 30 covers every returned article from each method.

Each result records the RRF score, source ranks, individual contributions,
original scores, PMID/text version, and the dense winning-window evidence when
that paper is in the dense source list. An absent dense window remains null,
rather than being invented for a lexical-only hit. Raw scores are diagnostic;
they do not enter the RRF formula, and RRF scores are not probabilities.

Reports retain both complete source runs, artifact/code hashes, configuration,
and a run ID. Corpus/model compatibility checks are inherited from dense search.
No saved articles, vectors, or relevance labels are rewritten.

## Four-method evaluation

```sh
python scripts/evaluate_hybrid.py benchmarks/pilot-v1/queries.jsonl \
  --qrels data/evaluation/pilot-v1/human-expanded-001.jsonl \
  --windows-root data/windows/pilot-v1-002 \
  --embeddings-root data/embeddings/pilot-v1 \
  --output results/evaluation/pilot-v1-rrf-001.json
```

The evaluator compares PubMed, BM25, dense, and RRF on the same candidates and
qrels. PubMed/BM25/dense top-10 rankings were checked against the prior expanded
baseline and are unchanged. The [recorded run](experiments/rrf-retrieval.json)
has the following provisional macro results:

| Method | nDCG@10 | Precision@10 | Judged coverage@10 |
| --- | ---: | ---: | ---: |
| PubMed | 0.578 | 0.93 | 1.00 |
| BM25 | 0.610 | 0.92 | 1.00 |
| Dense | 0.908 | 1.00 | 1.00 |
| RRF | 0.767 | 0.98 | 0.99 |

RRF is below dense on all nine fully judged questions. Its remaining question,
p08, introduces one unjudged pair (PMID 30597535). With depth 30, this paper can
enter RRF top-10 even though it was outside the separate BM25 and dense top-10
lists used to build earlier pools. Do not silently treat its missing label as a
human grade 0. Numeric metrics retain the evaluator's zero-contribution policy
and are marked provisional until the additional judgment is collected.

The initial equal-weight hybrid does not improve the strongest measured baseline.
Keep RRF as an experiment; dense remains the quality reference for subsequent
work. Do not optimize fusion weights or c against this tiny pilot and then call
the result held-out. Future reranking must be compared with dense alone as well
as with RRF: improving a weaker RRF baseline would not necessarily improve the
best available retrieval system. The planned Qwen experiment and possible
cross-encoder alternative remain subject to measured quality and latency gains.

The ten fusion calls took about 0.40 ms total, excluding the source retrieval,
query embedding, and report assembly. This is an observed component time, not
end-to-end latency or evidence of relevance quality.

## One additional judgment

```sh
python scripts/annotate_pilot.py build benchmarks/pilot-v1/queries.jsonl \
  --plan benchmarks/pilot-v1/plan.json \
  --retrieval-report results/evaluation/pilot-v1-rrf-001.json --method rrf \
  --qrels data/evaluation/pilot-v1/human-expanded-001.jsonl \
  --output data/annotation/pilot-rrf-supplement-001
python -m http.server 8766 --bind 127.0.0.1 --directory data/annotation
```

Open `http://127.0.0.1:8766/pilot-rrf-supplement-001/`. This packet contains only
one new card, with no source scores or ranks. The previous 199 judgments remain
unchanged. Download the annotation file, then use the existing `export` and
`merge` commands with this pool and the current base qrels. A complete merge will
produce 200 judged pairs in a new qrels file; rerun all four methods with that
file and a new output path. The older dense-only supplement commands remain
compatible through the `--dense-report` alias.

All 110 offline tests pass. New cases cover hand-calculated fusion scores,
one-based ranks, absence/empty lists, c=0, deterministic ties, invalid inputs,
score-scale independence, source compatibility, CLI replay, four-method
evaluation, and selecting only unknown RRF pairs for annotation. The real-model
hybrid CLI and ten-question evaluation also completed successfully.
