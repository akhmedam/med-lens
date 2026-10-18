# MedLens

Retrieval experiments over PubMed papers on depression and anxiety.

The project starts with BM25 and dense retrieval, then measures whether hybrid
ranking, LLM reranking, and full text help. Answers will link back to source
fragments. This is a literature search tool, not a diagnostic system.

## Status

Day 1: checking local inference on an M3 Pro with 36 GB unified memory.
Retrieval and answer generation are not implemented yet.

## Setup

Python 3.11 on Apple Silicon:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-macos-arm64.lock
python -m pip install --no-deps -e .
```

## Runtime check

```sh
python scripts/smoke_inference.py --output results/preflight.json
python scripts/smoke_inference.py --run --output results/inference.json
```

Run the lightweight output-validation tests with:

```sh
python -m unittest discover -s tests -v
```

The first runtime command checks MPS and the installed loading APIs without downloading
weights. The second first probes the Metal kernel, then downloads the official Qwen3.5-9B checkpoint, quantizes
supported linear layers to 4 bits, and runs two synthetic prompts on the same
model. It records versions, timings, output, and memory observations.

Model files stay in `.cache/` and are excluded from Git. Loading-time quantization
still downloads the original checkpoint. First-run timing includes downloads.

See [runtime notes](docs/experiments/runtime-smoke.md) for measured results and
[architecture decisions](docs/adr/0001-scope.md) for scope.
