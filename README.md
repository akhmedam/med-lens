# MedLens

A local biomedical literature search tool for macOS on Apple Silicon.

The supported application is one workflow: **fresh PubMed titles/abstracts →
BGE Dense ranking → exact evidence → Qwen3.5-4B/MLX source-findings cards**.
The primary interface is a simple terminal session. Cards contain source
quotations and applicability cautions, not a generated factual synthesis.
See the [MVP contract](docs/mvp.md) and [policy decision](docs/adr/0007-unified-mvp-policy.md).

## Status

The CLI and optional local HTTP API use the same pinned model, source-findings
format and bounded search policy. Defaults are 100 PubMed candidates and ten
ranked articles. Every question performs a fresh source search; unchanged
passage inputs reuse cached vectors. There is no additional reranker or
model/backend fallback.

English biomedical questions are the supported scope. Language and topic are
not classified, and inputs are not translated automatically. Nonempty search
results do not prove relevance or exhaustive PubMed coverage.

Exact source quotations are checked, but their scientific support and answer
completeness are not independently verified. Nonempty source-findings results
remain `partial`. Known selection/semantic limitations and missing independent
human references remain release conditions, as documented in the
[development assessment](docs/biomedical-quality.md),
[source-findings assessment](docs/biomedical-quality-improvement.md) and
[development plan](docs/development-plan.md) and [final validation](docs/final-validation.md). Historical human labels and later
AI annotations have separate provenance; neither is silently transferred to new
questions or source versions.

For the next independent round, the owner requires **at least 50% useful
selections across all twenty queries**, **at most 50% unnecessary abstentions
among human-confirmed answerable queries**, and **at most 60 seconds for each
complete request**. The [human-confirmation protocol](docs/human-confirmation.md)
records the new blank source-reference form and the later actual-output review.
The sixty-second deadline is an acceptance requirement; the current service does
not yet enforce that hard budget.

## Setup

Use Python 3.11 on macOS/Apple Silicon. The main environment and isolated MLX
runtime have separate locks:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-mvp-macos-arm64.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m venv .cache/venvs/mlx
.cache/venvs/mlx/bin/python -m pip install -r requirements-mlx-macos-arm64.lock
.venv/bin/python scripts/prepare_models.py --profile mvp --download --output results/setup-bge.json
.venv/bin/python scripts/prepare_mlx.py --download
```

The MVP asset profile prepares BGE and its saved tokenizer. `prepare_mlx.py`
prepares the pinned native Qwen4B model and official tokenizer. No original 9B,
MedCPT or Metal-kernel asset is required. Existing conflicting assets fail
verification rather than being overwritten. Use a fresh `--output` report name
when repeating asset preparation. See [native assets and provenance](docs/native-mlx.md).

The historical Day 1/Day 19 environment locks and the full research asset profile
are retained for experiments; they are not the supported application's setup.

## Ask a question

```sh
.venv/bin/medlens
```

Enter a biomedical research question in English. The CLI shows stage messages,
source findings with citations, and ranked PubMed articles. Commands are
`/help`, `/evidence`, `/export [PATH]` and `/quit`.

For one question instead of a session:

```sh
.venv/bin/medlens --question 'How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?'
```

`--search-only` returns articles and selected fragments without Qwen inference.
`--limit` and `--top-k` are bounded search settings; `--pubmed-query` supplies an
exact PubMed expression. The public CLI has no answer-style/model/backend switch.
[CLI usage](docs/interactive-cli.md) covers export, errors and graceful Ctrl+C.

## Local API

The optional API exposes the same policy on loopback HTTP, one job at a time:

```sh
.venv/bin/python -m pip install -r requirements-api.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/medlens-api
```

Submit JSON to `POST http://127.0.0.1:8000/v1/requests`, then poll its returned
`links.self`. `GET /health` remains responsive during processing. API
`answer_style`, if provided, must be `source-findings`; `draft` returns 422.
See the [request/result contract](docs/local-api.md).

## Local artifacts and moving machines

Git contains code, policies, compact reports and labeled evaluation artifacts.
Model weights, source snapshots, vectors and raw outputs remain local. Back them
up with the exact checkout/manifest; Git alone cannot restore them. A fresh
PubMed download has new source identities and cannot reconstruct the lost
historical pilot byte for byte.

The [old handoff](docs/handoff.md) and [bundle-loss recovery](docs/recovery-without-bundle.md)
pages preserve historical transfer/replay procedures. For a new MVP installation,
use the setup above. [MVP scope and remaining work](docs/mvp.md#research-and-remaining-work)
distinguish release gates from post-MVP portability, full text and scale work.

## Checks

The full offline regression suite also exercises historical retrieval tools.
In the prepared development environment, install API/test dependencies and the
pinned BM25 fixture dependency as well (BM25 is not an MVP runtime dependency):

```sh
.venv/bin/python -m pip install -r requirements-api.lock
.venv/bin/python -m pip install bm25s==0.3.14
.venv/bin/python -m unittest discover -s tests -v
```

Functional checks, exact citations and successful model execution do not replace
source-bound semantic assessment. Optional native-grammar checks use the
separate MLX environment. See the [frozen final results](docs/final-validation.md)
for current coverage, measured limits and unresolved release gates.

## Research commands and historical benchmarks

The commands below are research/replay tools. Their optional 9B, Transformers,
MedCPT, BM25/RRF, reranking and prose-draft paths are outside the supported
application. Research prose generation uses explicit `--answer-style draft` in
`ask_live.py` or `ask.py --generate`; frozen older experiment entry points retain
their original contracts. Their historical metrics do not describe the current
source-findings application. Original measurements and provenance remain in the
linked reports. Use a separate research environment with its recorded dependency
lock; the minimal MVP lock does not install every historical experiment. The
legacy `python` examples below assume that research environment is activated.

## Download papers

```sh
python scripts/pubmed_snapshot.py fetch \
  --query 'depression AND cognitive behavioral therapy' \
  --limit 30 --batch-size 10 \
  --output data/snapshots/depression-cbt
python scripts/pubmed_snapshot.py inspect data/snapshots/depression-cbt
```

`fetch` saves the raw ESearch JSON, EFetch XML batches, and a manifest with ordered
PMIDs, timestamps, request parameters, and hashes. Choose a new output directory
for each live search. `inspect` verifies a saved snapshot without contacting NCBI.
Batch size defaults to 200; the example uses 10 to exercise batching.

See the [ingestion check](docs/experiments/pubmed-ingestion.md) for a real run.

## Parse a snapshot

```sh
python scripts/parse_pubmed_snapshot.py build data/snapshots/depression-cbt \
  --output data/parsed/depression-cbt
python scripts/parse_pubmed_snapshot.py inspect data/parsed/depression-cbt
```

Parsing preserves abstract sections, contributors, date precision, publication
types, identifiers, and source provenance. See [record format and checks](docs/parsing.md).

## Search with BM25

```sh
python -m pip install -e '.[retrieval]'
python scripts/bm25_search.py data/parsed/depression-cbt \
  --query 'cognitive behavioral therapy depression' \
  --top-k 10 --output results/bm25/depression-cbt.json
```

Search runs offline over the selected dataset. Results include PMIDs, scores,
titles, and PubMed links. See [BM25 settings and checks](docs/bm25.md).

## Evaluate historical pilot retrieval

```sh
python scripts/evaluate_retrieval.py data/evaluation/queries.jsonl \
  --qrels data/evaluation/qrels.jsonl --split pilot \
  --output results/evaluation/pilot-001.json
```

This historical command does not evaluate the new biomedical annotation package.
The runner compares PubMed order and BM25 on the same candidates. Prepare query
and judgment files using the [evaluation format](docs/evaluation.md). Reports
include nDCG@10, binary metrics, and coverage; unjudged papers remain distinct
from irrelevant papers. The ten-query pilot uses the [v2 evaluation profile](docs/pilot-qrels-v2.md).

## Annotate the pilot

Ten questions and their PubMed search plan are saved in `benchmarks/pilot-v1/`.
The local annotation page combines PubMed/BM25 top-10 without showing method
names or ranks. See the [pilot workflow](docs/pilot.md) for data preparation,
annotation, export, and evaluation. The [original human-judged baseline](docs/pilot-baseline.md) is retained as history;
[v2](docs/pilot-qrels-v2.md) is the active judgment set.

## Prepare BGE windows

```sh
python -m pip install -e '.[tokenization]'
python scripts/prepare_windows.py tokenizer \
  --output .cache/tokenizers/bge-base-en-v1.5
python scripts/prepare_windows.py build data/parsed/pilot-v1/p01 \
  --tokenizer .cache/tokenizers/bge-base-en-v1.5 \
  --output data/windows/pilot-v1-002/p01
```

Section-bounded windows preserve source spans and fit BGE's input limit, including
the title. See [window format and checks](docs/embedding-windows.md). This prepares
inputs for the encoder below.

## Encode BGE windows

```sh
python -m pip install -e '.[embeddings]'
python scripts/embed_windows.py model --output .cache/models/bge-base-en-v1.5
python scripts/embed_windows.py build data/windows/pilot-v1-002/p01 \
  --model .cache/models/bge-base-en-v1.5 \
  --tokenizer .cache/tokenizers/bge-base-en-v1.5 \
  --output data/embeddings/pilot-v1/p01
```

The encoder saves normalized 768-dimensional vectors with their source-window
mapping. See [BGE settings and MPS checks](docs/bge-embeddings.md). Exact cosine
search is available below; see the [current pilot results](docs/pilot-qrels-v2.md).

## Search with dense retrieval

```sh
python scripts/dense_search.py data/parsed/pilot-v1/p10 \
  --windows data/windows/pilot-v1-002/p10 \
  --embeddings data/embeddings/pilot-v1/p10 \
  --query 'Does cognitive behavioral therapy reduce anxiety symptoms in children with anxiety disorders?' \
  --output results/dense/p10-001.json
```

Exact cosine search scores articles by their best window and returns that source
fragment. See [dense retrieval and pool expansion](docs/dense-retrieval.md) for
scoring and evaluation. The [current evaluation](docs/pilot-qrels-v2.md) uses 198
judged pairs and preserves two unknowns.

## Search with RRF

```sh
python scripts/hybrid_search.py data/parsed/pilot-v1/p08 \
  --windows data/windows/pilot-v1-002/p08 \
  --embeddings data/embeddings/pilot-v1/p08 \
  --query 'Does exposure-based cognitive behavioral therapy reduce panic symptoms in adults with panic disorder?' \
  --output results/hybrid/p08-001.json
```

The initial hybrid combines BM25/dense ranks with equal weights, c=60, and depth
30. It is an experiment: [pilot results and fusion policy](docs/hybrid-retrieval.md)
show no gain over dense. The [v2 comparison](docs/pilot-qrels-v2.md) is current;
the [original four-method baseline](docs/rrf-baseline.md) is historical.

## Rerank with Qwen

```sh
python scripts/evaluate_reranking.py benchmarks/pilot-v1/queries.jsonl \
  --qrels data/evaluation/pilot-v1/pilot-qrels-v2.jsonl \
  --baseline results/evaluation/pilot-v1-v2-retrieval-001.json \
  --output results/evaluation/pilot-v1-v2-qwen-new-run-001.json
```

To rescore the existing outputs without running Qwen again, use the
[active profile replay](docs/pilot-qrels-v2.md#replay-the-current-evaluation). The command above
starts a new inference run after that replay has prepared the v2 retrieval baseline.
The runner uses one cached model to reorder dense and RRF top-10 lists separately.
Invalid output preserves the source order. See the [prompt and runtime policy](docs/llm-reranking.md)
and [current pilot comparison](docs/pilot-qrels-v2.md).

## Prepare evidence for an answer

```sh
HF_HUB_OFFLINE=1 python scripts/prepare_evidence.py \
  --output results/evidence/pilot-v1-dense-qwen-004.json \
  --summary results/evidence/pilot-v1-dense-qwen-summary-004.json
```

This uses frozen pilot article rankings, stored windows/vectors and the Qwen
tokenizer to select up to six source excerpts within 4000 evidence tokens.
Citation IDs map to exact source spans. See the
[selection policy and checks](docs/evidence-context.md).

## Generate cited drafts

`scripts/generate_answers.py` is a frozen 9B experiment; its model and saved
protocol do not switch with the current live/local CLI default.

```sh
python scripts/generate_answers.py results/evidence/pilot-v1-dense-qwen-004.json \
  --include-controls \
  --output results/generation/pilot-v1-cited-004.json \
  --summary results/generation/pilot-v1-cited-summary-004.json
```

The generator selects sentence IDs; code supplies exact quotations and PubMed
links. Unsupported or unclear claims are withheld by a second call to the same
Qwen instance. Model review can miss scope errors, so every displayed answer is
an unverified research draft. Invalid or truncated output shows no answer.
Choose fresh output paths. See the [protocol and pilot results](docs/cited-answers.md).
This benchmark runner consumes saved contexts. Use `scripts/ask.py` above for
a fresh search-to-answer request against a local collection.

## MedCPT ranking experiments

```sh
HF_HUB_OFFLINE=1 python scripts/medcpt_search.py data/parsed/pilot-v1/p01 \
  --windows data/windows/pilot-v1-002/p01 \
  --mode medcpt \
  --query "How does cognitive behavioral therapy compare with antidepressants for depressive symptoms in adults with major depression?" \
  --output results/ranking/medcpt-example.json
```

For Dense top-10 → MedCPT, use `--mode dense-medcpt` and add
`--embeddings data/embeddings/pilot-v1/p01`. Both modes score every window of each
eligible article and aggregate by maximum logit. Direct MedCPT needs no BGE
model or embeddings at query time. See the [ranking contract and pilot
preparation](docs/medcpt-ranking.md) and the [quality comparison](docs/medcpt-evaluation.md).
The primary workflow omits `--rerank`; historical Qwen reranking remains an
explicit 9B comparison.

## MedCPT cross-encoder probe

```sh
python scripts/probe_medcpt.py --download \
  --output results/preflight/medcpt-example.json \
  --summary results/preflight/medcpt-example-summary.json
```

This pins and checks MedCPT assets, audits full question/document token budgets,
and measures CPU/MPS batching on pilot source windows. The article-ranking
commands above build on this scorer. See the [scorer contract and
measurements](docs/medcpt.md) and the [frozen held-out protocol](benchmarks/heldout-v1/README.md).

## Historical 9B runtime check

These smoke commands exercise the frozen Transformers/Metal 9B experiment.
They do not test the current 4B default; use the native setup and request above
for that workflow.

```sh
python scripts/smoke_inference.py --output results/preflight.json
python scripts/smoke_inference.py --run \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --output results/inference.json
```

Run the offline tests with:

```sh
python -m unittest discover -s tests -v
```

The first runtime command checks MPS without downloading weights. The second
probes the Metal kernel, loads Qwen3.5-9B, quantizes supported linear layers to
4 bits, and runs two synthetic prompts on the same model. It records versions, timings, output, and memory observations.

Model files stay in `.cache/`; snapshots stay in `data/`. Both are excluded from
Git. The historical Transformers smoke can download the original 9B checkpoint
before loading-time quantization; its first-run timing includes those downloads.
The current 4B setup is described separately above.

See [runtime notes](docs/experiments/runtime-smoke.md) for measured results and
[architecture decisions](docs/adr/0001-scope.md) for scope.
