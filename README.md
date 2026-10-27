# MedLens

Retrieval experiments over PubMed papers on depression and anxiety.

The project starts with BM25 and dense retrieval, then measures whether hybrid
ranking, LLM reranking, and full text help. Answers will link back to source
fragments. This is a literature search tool, not a diagnostic system.

## Status

PubMed downloads, verified snapshot replay, structured XML-to-JSONL parsing, and
article-level BM25 search work. Local Qwen3.5-9B inference runs through
Transformers/MPS with 4-bit Metal linear layers. Dense retrieval and evidence-grounded answers are next.

## Setup

Python 3.11:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

PubMed ingestion uses the standard library. For the local model on Apple Silicon:

```sh
python -m pip install -r requirements-macos-arm64.lock
```

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
from irrelevant papers. The ten-query pilot now has [human-judged baseline results](docs/pilot-baseline.md).

## Annotate the pilot

Ten questions and their PubMed search plan are saved in `benchmarks/pilot-v1/`.
The local annotation page combines PubMed/BM25 top-10 without showing method
names or ranks. See the [pilot workflow](docs/pilot.md) for data preparation,
annotation, export, and evaluation. The first [human-judged baseline](docs/pilot-baseline.md) is complete.

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
search is available below; the expanded pilot now has [human-judged results](docs/dense-baseline.md).

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
scoring and evaluation. All 199 pilot pairs are now judged; see the
[three-method baseline](docs/dense-baseline.md).

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
show no gain over dense, with one additional judgment still needed.

## Runtime check

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

Model files stay in `.cache/`; snapshots stay in `data/`. Both are excluded from Git. Loading-time quantization
still downloads the original checkpoint. First-run timing includes downloads.

See [runtime notes](docs/experiments/runtime-smoke.md) for measured results and
[architecture decisions](docs/adr/0001-scope.md) for scope.
