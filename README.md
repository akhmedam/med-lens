# MedLens

Retrieval experiments over PubMed papers on depression and anxiety.

The project starts with BM25 and dense retrieval, then measures whether hybrid
ranking, LLM reranking, and full text help. Experimental answer drafts link back
to exact source quotations. This is a literature search tool, not a diagnostic
system.

## Status

PubMed ingestion, verified snapshot replay, and structured XML parsing work.
Search supports BM25, exact dense retrieval, and RRF. The recorded historical pilot uses
`pilot-qrels-v2`: 198 of 200 query/article pairs are judged. Historical Qwen3.5-9B experiments
rerank frozen top-10 lists through
Transformers/MPS with 4-bit Metal linear layers. See the [current evaluation](docs/pilot-qrels-v2.md)
for metrics and annotation provenance. Saved evidence contexts can feed
experimental cited answer drafts with a separate model support review. These
drafts require independent review; exact quotations do not prove claim support.
A [live request](docs/live-search.md) now searches PubMed and downloads current
abstracts for each question, reuses unchanged passage vectors, and runs Dense,
source selection and Qwen generation/review. The user-selected development default
is **Qwen3.5-4B in native MLX**, with short draft instructions and named axes-first
support review. [Native inference](docs/native-mlx.md) runs in an isolated Apple
Silicon environment. `--answer-model 9b` selects the explicit comparison; the
Transformers/Metal comparison also requires `--engine transformers`.
Local-collection replay remains available.
The [expanded MedCPT comparison](docs/medcpt-evaluation.md) now uses 210 pairs
(208 judged). Neither MedCPT mode improves pilot nDCG over Dense; the comparison
remains provisional because two pairs are unjudged.
The [M3 Pro cost comparison](docs/retrieval-cost.md) covers 10/20/30-article pools
and MedCPT depths: cached Dense is fastest; direct MedCPT avoids passage-embedding
construction for fresh windows. The primary chain is **Dense → exact evidence
→ Qwen draft → same-model support review**. [ADR 0005](docs/adr/0005-qwen4b-development-default.md) records the new 4B
development default; the original 9B choice remains historical in ADR 0003.

The user supplied the new blind human labels. Both reviewers failed the pilot
safety gate: 4B admitted 3/5 human comparison-mismatched claims and 9B admitted
4/5. Human answer usability was 5/6 for 4B and 4/6 for 9B, but the all-context
4B usability gate still failed. Six related source pools cover only two
independent questions. The development choice does not establish clinical or
general biomedical equivalence; every generated answer remains a research draft
requiring human review.

The [historical semantic checks](docs/day21-semantic-reliability.md) document
remaining source-support errors. The seven-axis v5 candidate regressed on
controlled development cases and remains opt-in through the legacy route. The
4B default uses the v4 taxonomy with named consistency constraints. Neither
reviewer establishes clinical correctness.

The [shared local corpus](docs/shared-corpus.md) pools the rebuilt topic collections
into 277 unique articles and 850 windows, with an immutable embedding cache.
Measured preparation and search costs cover the start of the Small → Medium
transition; retrieval and clinical-answer quality are not revalidated by pooling.

A six-article [PMC full-text feasibility study](docs/pmc-fulltext.md) compares
paired abstract and narrative indexes. Full text adds source coverage and changes
rank order; the pilot has no independent relevance labels, so abstract retrieval
remains primary.

The accepted [development plan](docs/development-plan.md) covers Days 15–28,
including the two MedCPT experiments and an interactive CLI as the primary UI.

## Moving to another Mac

Follow the [handoff guide](docs/handoff.md) to clone the code, restore the checked
local-artifact bundle and rebuild Python 3.11 dependencies. Pilot labels and
provenance are tracked; model weights, primary texts, vectors and raw runs are
transferred separately. The [pipeline decision](docs/pipeline-options.md) is
accepted for its retrieval/generation chain; the current model default is 4B
under [ADR 0005](docs/adr/0005-qwen4b-development-default.md). If the archive is lost,
follow [recovery without a bundle](docs/recovery-without-bundle.md). Pinned assets
and a separate 277-paper/909-window corpus were recovered on the Air M3, 24 GiB;
see the [Air recovery record](docs/experiments/air-recovery.json) for measured
runtime scope and format failures. Historical raw experiments remain lost.

## Setup

Python 3.11:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

PubMed ingestion uses the standard library. For the local model on Apple Silicon:

```sh
python -m pip install -r requirements-macos-arm64-day19.lock
```

The request CLIs default to the native 4B profile. Prepare the isolated runtime
and pinned assets:

```sh
python -m venv .cache/venvs/mlx
.cache/venvs/mlx/bin/python -m pip install -r requirements-mlx-macos-arm64.lock
python scripts/prepare_mlx.py --download
```

This setup needs no 9B weights. See the [native runtime guide](docs/native-mlx.md)
for local asset reuse, provenance, and `prepare_mlx.py --model 9b` conversion.
Use `--answer-model 9b` for native 9B, or add `--engine transformers` for the
original 9B Metal backend. There is no automatic model/backend fallback.

## Ask a question

Run the selected live PubMed → Dense → exact evidence → Qwen4B draft/review workflow:

```sh
python scripts/ask_live.py \
  --query "Does cognitive behavioral therapy reduce anxiety symptoms in adults with generalized anxiety disorder?" \
  --limit 100 --output-dir results/requests/live-gad-example
open results/requests/live-gad-example/report.html
```

Generation is enabled by default; `--search-only` runs diagnostic retrieval and
excerpts. Every request searches and fetches PubMed anew; the embedding cache
reuses unchanged inputs. The current English question scope is depression and
anxiety/panic. PubMed candidate collection is bounded (1–300, initial default 100),
and the exact submitted query and PubMed translation are recorded in the report.
This does not search every PubMed article semantically. See the
[live request guide](docs/live-search.md) for query overrides, timing and limits.
Use a fresh output directory for each run.

For offline replay, `scripts/ask.py` still accepts dataset/window/embedding
directories; pass `--generate` and omit `--rerank` for the same answer pipeline.
See the [local request guide](docs/local-requests.md) and
[shared-corpus guide](docs/shared-corpus.md).

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

## Evaluate retrieval

```sh
python scripts/evaluate_retrieval.py data/evaluation/queries.jsonl \
  --qrels data/evaluation/qrels.jsonl --split pilot \
  --output results/evaluation/pilot-001.json
```

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
