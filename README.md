# MedLens

MedLens searches PubMed for an English biomedical question, ranks papers locally,
and presents findings as exact source quotations with citations and study context.
It provides an interactive terminal and a local HTTP API.

This is a research tool. Source selection can make scientific or scope errors;
results require human review and are not clinical advice. See the
[current assessment](docs/usefulness-research.md).

## Prerequisites

- macOS on Apple Silicon and Python 3.11.
- Git and internet access to PubMed and Hugging Face.
- Disk space for the Python environments, BGE and Qwen3.5-4B model weights,
  downloaded papers and caches.

## Installation

```sh
git clone https://github.com/akhmedam/med-lens.git
cd med-lens
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-arm64.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m venv .cache/venvs/mlx
.cache/venvs/mlx/bin/python -m pip install -r requirements-mlx-macos-arm64.lock
.venv/bin/python scripts/prepare_models.py --download --output results/setup-bge.json
.venv/bin/python scripts/prepare_mlx.py --download
```

Both environments are required: the main process runs retrieval and the API;
the isolated MLX process runs Qwen. Preparation downloads pinned assets and checks
their hashes. See [model setup](docs/native-mlx.md) for details and recovery.

## Usage

```sh
.venv/bin/medlens
```

Enter a question in English. Use `/help`, `/evidence`, `/export [PATH]` and `/quit`.
For a single question:

```sh
.venv/bin/medlens --question 'What are the top 5 lifestyle factors for mental health?'
```

Start the local API with `.venv/bin/medlens-api`.
See the [CLI guide](docs/interactive-cli.md) and [API contract](docs/local-api.md).

## How it works

1. Fetch fresh PubMed titles and abstracts, with up to 100 candidates by default.
2. Rank article passages with BGE embeddings and return the top ten papers.
3. Use Qwen3.5-4B through MLX to select source findings, then check quotation and
   citation identity. Scientific relevance and completeness still require review.

Unchanged passages reuse cached embeddings. Models, caches, source snapshots and
request outputs stay in local directories excluded from Git.

Code lives in `src/medlens/`, tests in `tests/`, model pins in `configs/`, and
research utilities in `scripts/`. See [CONTRIBUTING.md](CONTRIBUTING.md) for checks,
the [MVP contract](docs/mvp.md) for supported behavior, and the
[development plan](docs/development-plan.md) for remaining work.
