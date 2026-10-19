# MedLens

Retrieval experiments over PubMed papers on depression and anxiety.

The project starts with BM25 and dense retrieval, then measures whether hybrid
ranking, LLM reranking, and full text help. Answers will link back to source
fragments. This is a literature search tool, not a diagnostic system.

## Status

PubMed downloads and verified offline snapshot replay work. Local Qwen3.5-9B
inference runs through Transformers/MPS with 4-bit Metal linear layers.
BM25, dense retrieval, and evidence-grounded answers are next.

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
