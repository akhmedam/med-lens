# Retrieval evaluation

The active ten-question pilot profile is [pilot-qrels-v2](pilot-qrels-v2.md).
The file formats and metric definitions below apply to every annotation version.

The runner compares the saved PubMed candidate order with BM25 on the same
verified parsed dataset for each query. It makes no network calls. Different
queries may point to different candidate datasets; results are averaged by query.

```sh
python -m pip install -e '.[retrieval]'
python scripts/evaluate_retrieval.py data/evaluation/queries.jsonl \
  --qrels data/evaluation/qrels.jsonl --split pilot \
  --output results/evaluation/pilot-001.json
```

Inputs are described below; create them for the saved datasets you want to judge.
The default split is `pilot`, cutoff is 10, and BM25 parameters are `k1=1.5`,
`b=0.75`. Override with `--split`, `--k`, `--k1`, and `--b`. Existing output files
are refused. Invalid inputs or failed dataset verification produce no report.

## Query and judgment files

Both files are UTF-8 JSONL, one object per line. A query row has exactly these fields:

```json
{"query_id":"pilot-001","text":"cognitive behavioral therapy depression","split":"pilot","group_id":"cbt-depression","dataset":"../parsed/depression-cbt"}
```

`dataset` is relative to the query file's directory; absolute paths also work.
`split` is `pilot`, `dev`, or `test`. Query IDs must be unique. Paraphrases share a
`group_id`; a group crossing splits is rejected. Only the selected split's
datasets are loaded and ranked. The runner validates all input rows, but does not
include held-out queries or their judgments in the selected split's output.

A judgment row has exactly these fields:

```json
{"query_id":"pilot-001","pmid":"38724122","relevance":null}
```

This example is deliberately unjudged. Replace `null` with a relevance judgment,
recording its provenance separately:

| Grade | Meaning |
| --- | --- |
| 2 | Directly addresses the question and its stated population/intervention/outcome constraints |
| 1 | Useful partial evidence or background, but does not directly address the whole question |
| 0 | Does not provide relevant evidence for this question |
| null / absent pair | Not judged |

Relevance is query-specific, not a universal paper quality or treatment efficacy
score. Judge what the available title and abstract support; do not infer findings
from publication type alone. Keep assessment notes separately if a decision needs
explanation. Query intent and rubric should be settled before annotation.

Unknown query IDs, duplicate query/PMID pairs, and grades outside `0/1/2/null` are
errors. Relevance must be an integer, not a Boolean or floating-point number.
Judgments for known relevant papers outside the current candidates are allowed.
Keep each judgment file with the dataset versions it was annotated against;
the runner records exact input and dataset hashes but cannot detect an annotator
reusing a stale label after a source text changes.

## Metric definitions

The primary metric is **nDCG@10**. Binary metrics count grades 1 and 2 as relevant.

| Metric | Definition |
| --- | --- |
| nDCG@k | DCG uses gain `2^rel - 1` and discount `log2(rank + 1)`; ideal DCG sorts all judged grades for the query, including papers outside candidates |
| Precision@k | Relevant hits in the first k results divided by k, even if fewer results were returned |
| Recall@k | Relevant hits divided by all known relevant judgments for the query |
| MRR@k | Reciprocal rank of the first relevant hit within k, or zero when none is retrieved; macro average gives MRR |
| Judged coverage@k | Judged results divided by actual returned results within k |
| Candidate coverage | Known relevant judgments present in the candidate set divided by all known relevant judgments |
| Candidate judged coverage | Judged candidates divided by candidate count |

Unjudged papers contribute zero to pooled ranking metrics; their labels remain
`null`, with `judged=false` in the ranking. This convention can penalize methods
retrieving unjudged papers. Always inspect coverage alongside metrics. Missing
result slots are not counted as unjudged papers; `returned_count` distinguishes
a short list from a full top-k.

No positive judgments means nDCG, Recall, MRR, and candidate coverage are undefined
(`null`). Precision is also undefined when there are no judgments at all; with
only grade-0 judgments it is zero. An empty ranking has zero retrieval scores if
positive judgments exist, but undefined judged coverage. Macro means exclude
undefined values and show `defined_queries` and `total_queries` for each metric.

Recall and candidate coverage are relative to **known judgments**, not all
relevant literature in PubMed. A top-10 annotation pool cannot measure absolute
recall. Freeze and version the pool when comparing methods. The planned pool
unites unique PMIDs from PubMed/BM25/dense/RRF top-10 and hides method names and
ranks during annotation. Dense and RRF do not exist yet; the runner currently
compares only PubMed order and BM25.

## Reports and checks

Reports include per-query rankings, labels, metrics, a full BM25 run, macro means,
input hashes, dataset IDs, evaluation/retrieval code hashes, and package versions.
The run ID hashes the inputs, split, cutoff, metric policy, and BM25 run IDs.
Repeated evaluation of the same saved inputs has the same ID and scores; timing
observations may differ. BM25 timings exclude loading and verification. Memory
and end-to-end latency benchmarks are not part of this runner yet.

The [Day 5 check](experiments/retrieval-evaluation.json) records two separate runs:
fictional records with synthetic labels to exercise graded metrics, and our
30-paper dataset with no labels to verify that quality remains undefined. The
synthetic scores say nothing about biomedical retrieval quality.

All 69 offline project tests pass. New cases check hand-computed scores, short
and empty rankings, unjudged papers, relevant papers outside candidates, group
leakage, input validation, deterministic replay, and the CLI. Day 6 will prepare
the ten-query pilot and human annotation; this change supplies the measurement
tools without manufacturing a benchmark.
