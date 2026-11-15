# Supported MVP

This page and [ADR 0007](adr/0007-unified-mvp-policy.md) define the supported
application. Older experiment reports preserve their original models, options
and measurements; they do not select the current application policy.

| Decision | Supported application |
| --- | --- |
| Interface | Plain-text `medlens` CLI; optional loopback `medlens-api` |
| Platform | macOS on Apple Silicon; Python 3.11 |
| Question scope | English biomedical research questions |
| Acquisition | Fresh bounded PubMed title/abstract search for every request |
| Candidates and display | Default 100 candidates and top-10 articles; request limits 1–300 and 1–10 |
| Ranking | Pinned BGE-base-en-v1.5 on MPS; maximum-window cosine similarity per article |
| Answer format | Source-findings cards: exact study-context and finding quotations |
| Generator | Pinned Qwen3.5-4B four-bit artifact in the isolated MLX runtime |
| Reranking and fallback | No additional reranker; no automatic model/backend or stale-source fallback |
| Verification | Exact quotation/source binding; no independent semantic or clinical verification |

## Run

Follow [README setup](../README.md#setup), then start the session:

```sh
.venv/bin/medlens
```

For one question instead of a session:

```sh
.venv/bin/medlens --question 'How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?'
```

`/help`, `/evidence`, `/export [PATH]` and `/quit` are the terminal commands.
`--search-only` disables Qwen generation and retains articles and selected
fragments. The [CLI guide](interactive-cli.md) describes bounded search settings,
public HTML/JSON export, exit codes and graceful shutdown. The
[API guide](local-api.md) documents the same workflow over local HTTP.

The public CLI has no answer-style, model or inference-backend selector. API
`answer_style`, if supplied, must equal `source-findings`; `draft` is rejected.
This is one supported answer format, not an automatic choice between pipelines.

## What the result means

Each request searches PubMed again. The original question drives ranking and
source selection. A transparent lexical planner removes recorded question
scaffolding for candidate acquisition; an explicit `--pubmed-query` expression
is submitted unchanged. Language and topic are not classified, and no automatic
translation is performed. Nonempty results do not establish relevance or
exhaustive coverage of PubMed.

Unchanged passage inputs reuse verified cached BGE vectors. New or changed
inputs are encoded. Requests retain source snapshots and exact text versions;
an upstream failure is not replaced with an older successful collection.

Qwen selects source quotations rather than writing a factual synthesis. Cards
retain source titles, study context, findings, citations and applicability
cautions. A nonempty answer remains `partial`: exact quotation identity does
not certify support for the question or completeness. `independent_validation`
and semantic-support verification remain false. Empty evidence and scope
exclusions produce an explicit insufficient-evidence result; malformed output
is withheld. Execution status and answer status are separate.

One process runs one accepted job at a time. BGE and Qwen phases are sequential.
The runtime is released before the next terminal question is accepted. Ctrl+C
during work requests exit after that job and cleanup, rather than hard
cancellation. Public reports exclude raw prompts, runtime paths and diagnostic
exceptions; private run artifacts remain local.

## Research and remaining work

`scripts/ask_live.py` and `scripts/ask.py` remain research/replay tools with
explicit experimental options. For their older prose draft/review path, pass
`--answer-style draft` explicitly. Their `--answer-model`, `--engine`, CPU BGE
and reranking choices are not supported application options. Frozen 9B scripts
and historical benchmark records are not silently converted to the MVP policy.

Windows/Linux, CUDA, CPU-only Qwen, other model/backend profiles, PMC full text,
a larger measured Medium corpus, ANN infrastructure and a web interface are
outside this MVP. Their existing experiments remain available as research.

Known selection omissions, uncertain source applicability and unreliable
semantic-relation judgments remain limitations. The tuned development AI
assessment is not an independent human reference. Final source-bound quality
assessment, disclosed reference coverage and unresolved defects remain release
conditions; changing the interface does not satisfy them. Clean model setup
and dependency/CI checks remain separate release work.

The [final confirmation](final-validation.md) passed 12 operational cases but did
not accept semantic quality for release. AI diagnostics identified concrete
scope mismatches; finalized independent references are still absent. The held-out
outputs have now been observed. Future behavior changes require another untouched
confirmation set. Parent RSS growth over this short series is also disclosed;
long-session retention must be investigated independently of native cleanup.
