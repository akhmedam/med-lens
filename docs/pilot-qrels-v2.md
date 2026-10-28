# Pilot qrels v2

`pilot-qrels-v2` is the active relevance-judgment set for the ten-question pilot.
The previous human annotations and all reports computed from them remain historical
artifacts. Changing judgments changes evaluation scores, not retrieval, model
weights, fusion parameters or the saved Qwen permutations.

## Annotation provenance

Codex prepared these assessments by reading the frozen titles and complete primary
abstracts, with a rationale and confidence flag for each pair. They are AI-generated
judgments, not expert-human ground truth or an independent blinded evaluation.
The annotator had prior conversation context, including aggregate retrieval results.
The user selected this version as the working pilot set. The Qwen model being
evaluated was not run as the grader.

There are 200 pairs: 100 grade 2, 71 grade 1, 27 grade 0, and two `null` judgments.
The two papers without abstracts were left unknown rather than guessed. Age ranges,
condition definitions, intervention details and measured outcomes were considered
against each question's original relevance intent. Missing explicit adult ages are
flagged in the local notes; some judgments remain debatable. The original p09
intent allows other-age evidence as partial, whereas some other questions enforce
age exclusions. The rubric was not harmonized retrospectively for this version.

Individual labels, rationale text and full reports stay in the ignored local
`data/` and `results/` directories. The committed profile records paths, hashes,
counts and annotation provenance. The local review page is
`data/annotation/pilot-ai-review-001/index.html`.

## Current results

Macro metrics over the same ten questions and frozen candidate/reranking lists:

| Method | nDCG@10 | Precision@10 | MRR@10 | Judged coverage@10 |
| --- | ---: | ---: | ---: | ---: |
| PubMed | 0.631923 | 0.820000 | 0.950000 | 0.990000 |
| BM25 | 0.633636 | 0.840000 | 0.933333 | 1.000000 |
| Dense | 0.871333 | 0.950000 | 1.000000 | 0.990000 |
| RRF | 0.795642 | 0.920000 | 0.950000 | 1.000000 |
| Dense + Qwen | 0.891941 | 0.950000 | 1.000000 | 0.990000 |
| RRF + Qwen | 0.842411 | 0.920000 | 1.000000 | 1.000000 |

These scores are provisional: unknown judgments contribute zero numerically but
remain `null`, and coverage is reported explicitly. Dense + Qwen remains above
dense, which remains above RRF + Qwen and RRF. The small BM25/PubMed difference
should not be treated as a reliable ordering. Scores from different qrels versions
are not evidence of an improvement or regression in the search implementation.

The Qwen outputs and runtime observations are the same ones measured in the
[original reranking experiment](qwen-reranking-baseline.md): median 25.73 seconds
per call. No model was rerun for this rescore. Reranking remains opt-in; a small
AI-judged development pilot does not settle the quality/latency trade-off or
replace broader human and held-out validation.

## Replay the current evaluation

Run from the repository root on the original checkout with the frozen local
artifacts available:

```sh
python scripts/rescore_pilot.py \
  --output results/evaluation/pilot-v1-v2-rescore-001.json \
  --retrieval-output results/evaluation/pilot-v1-v2-retrieval-001.json \
  --summary docs/experiments/pilot-qrels-v2.json
```

Existing paths are refused; choose fresh paths for a repeated replay, including
the summary path. The default profile is
[`benchmarks/pilot-v1/evaluation.json`](../benchmarks/pilot-v1/evaluation.json).
It pins the current queries/qrels and the original source qrels/retrieval/Qwen
reports by SHA-256. The old qrels are used only to verify the historical metrics
before applying v2. No network or encoder/LLM inference is involved.

Replay checks dataset identities, query membership, source metric values and
Qwen output permutations/fallbacks. It writes a six-method local report, a
rescored four-method retrieval baseline usable by `evaluate_reranking.py`, and
a public aggregate summary with no individual judgments or article text.

For a new retrieval experiment, pass
`--qrels data/evaluation/pilot-v1/pilot-qrels-v2.jsonl` explicitly. For new Qwen
inference, the retrieval baseline must have the same query/qrels hashes; mixed
versions are rejected. The profile selects the current version for replay and
documents the convention; existing CLIs still take explicit inputs.

The raw judgments are not bundled with a fresh public clone. Without the local
artifacts, inspect the [aggregate report](experiments/pilot-qrels-v2.json) or
prepare a separately versioned annotation set for your own corpus.
