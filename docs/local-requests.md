# Local search-to-answer requests

For a fresh PubMed search on each question, use the primary
[live request command](live-search.md). This page documents prepared-collection
requests for offline replay and controlled experiments.

`scripts/ask.py` runs a fresh English question against one verified local
collection. It saves a human-readable `report.html` and a full `run.json` in a
new directory. No pilot relevance labels or saved rankings enter this path.
The primary workflow uses Dense followed by native Qwen3.5-4B generation and
same-model support review: pass `--generate` and omit `--rerank`. The
[development-default decision](adr/0005-qwen4b-development-default.md) supersedes
the earlier 9B default while preserving the retrieval/source architecture.

## Run

From the project root, with the existing virtual environment and cached models:

```sh
.venv/bin/python scripts/ask.py data/parsed/recovery-20261106/p06 \
  --windows data/windows/recovery-20261106/p06 \
  --embeddings data/embeddings/recovery-20261106/p06 \
  --query "Does cognitive behavioral therapy reduce anxiety symptoms in adults with generalized anxiety disorder?" \
  --generate --output-dir results/requests/gad-answer-example
open results/requests/gad-answer-example/report.html
```

The CLI without `--generate` is a diagnostic search/excerpt mode; generation
remains a required part of the primary product workflow.
`--generate` enables experimental answer drafts; `--rerank` enables optional
Qwen reranking of the dense results. These flags work independently. `--top-k`
accepts one to ten articles and defaults to ten. `--device cpu` changes BGE only;
Qwen generation and reranking require Apple GPU execution. The default is the
[isolated native MLX 4B profile](native-mlx.md). Pass `--answer-model 9b` for native
9B or `--answer-model 9b --engine transformers` for the original MPS comparison.
Preparation of the default profile does not require 9B weights.

Search covers only the supplied dataset. The frozen pilot's topic collections
each contain at most thirty papers, selected by earlier PubMed queries. A new
question does not expand that candidate set. The report displays collection size
and scope; similarity scores are not probabilities of relevance. No cosine
threshold or off-topic detector is implemented. The same 4000-token evidence
budget and six-fragment selection policy apply to every question; their
outcome-section preference was developed for treatment-efficacy questions.

## Request lifecycle

1. Validate the dataset, windows and embedding artifacts, then encode the new
   question with BGE once. Rank articles by maximum-window cosine similarity.
2. Release the request's BGE instance before loading Qwen. Pack dense evidence
   before optional reranking so a model failure still leaves source passages.
3. If requested and at least two hits exist, load Qwen once and rerank. Invalid
   permutations preserve the dense order. Repack evidence when the order changes.
4. If requested, use that same instance to generate and review a cited draft.
   Empty contexts abstain without a model call. Otherwise the generator and
   strict source/semantic gates remain active. The default compact policy uses
   `cited-answer-v7` and six named `claim-support-v4` review fields with
   `axes-first-v1`. Logical review consistency and exact citations do not establish
   factual support. Explicit 9B comparisons retain the legacy v6 policy.

The default mode loads BGE and the cached Qwen tokenizer, but no Qwen weights.
Retrieval/tokenizer-only operation can run with `HF_HUB_OFFLINE=1`. Native MLX
uses the prepared local artifact. The explicit Transformers comparison may need
network access for its Metal-kernel publisher/version check even though its
model weights load from the local cache.

Every completed stage checkpoints both files using atomic replacement. A run
owns a new directory and cannot overwrite an existing request. If model loading
or inference fails, completed search and context data remain in the report.
The JSON records the failing stage and error. A forced process kill can leave a
`running` checkpoint; it must not be interpreted as a completed run.

Execution status and answer status are separate. `complete` means the pipeline
finished, even if a draft failed validation. Exit codes are:

| Code | Meaning |
| --- | --- |
| 0 | Completed search; any requested answer passed its format gates or abstained |
| 1 | Setup, artifact, model or inference error |
| 2 | CLI argument error, or a completed run whose draft failed validation; inspect `run.json` to distinguish |
| 130 | Interrupted request with its last completed stages retained |

Existing output directories are rejected before any inference. The HTML uses
escaped text and code-generated PubMed links, with no scripts or remote assets.
Search results, selected excerpts and model claims have separate sections.
Visible generated claims are labeled research drafts requiring independent
review. A format failure suppresses the answer while keeping the search useful.

## Historical 9B validation

The observations below predate the 4B default and use the original 9B runtime.
Current 4B integration validation is [recorded separately](experiments/qwen4b-default-integration.json).

The public [integration record](experiments/local-requests.json) contains timings,
artifact identities and source checks, without article text or raw model output.
These are smoke requests against the small pilot collections, not a new ranking
or clinical-answer benchmark.

| Local collection and mode | Request time | Result |
| --- | ---: | --- |
| p01: search and excerpts | 3.34 s | Ten articles, six excerpts; no Qwen model loaded |
| p06: search, rerank and generate | 106.05 s | Three draft claims; one Qwen load, three calls |
| p03: search and generate | 73.64 s | Invalid JSON in support review; answer withheld, search retained |

These single-run times start after output-directory creation and include artifact
and tokenizer loading, BGE, any Qwen load/inference, and intermediate checkpoints.
They exclude Python startup/imports and the last report write. Reranking time
includes the Qwen load when it is the first model stage; generation time includes
it otherwise. Do not add model load time again.

All eighteen selected excerpts and five visible claim citations matched the
stored source spans. All eighty-two catalog entries resolved. Offline cache replay
reproduced the two generation results without inference; implementation hashes
and collection identities matched. Browser inspection covered the search-only,
visible-draft and validation-failure reports. The [Day 13 findings](cited-answers.md) still apply:
same-model review misses population, comparison and subgroup errors.

Unit tests cover one runtime across all three Qwen calls, fallback ordering,
empty evidence without Qwen, preserved search/context after model failure,
withheld malformed drafts, tokenizer mismatch, HTML escaping and output reuse
protection. Raw request files stay under ignored `results/requests/`.

The [shared local corpus](shared-corpus.md) now supports the same CLI with its
`dataset`, `windows` and `embeddings` directories. It starts the Small → Medium
transition at 277 unique articles; no new-paper fetching occurs during a request.
A license-gated PMC full-text experiment is documented in [pmc-fulltext.md](pmc-fulltext.md).
Abstract retrieval remains the primary workflow until full-text relevance and
answer support receive independent assessment. A served API also remains open work.

## Historical 9B Air recovery

The [recovery guide](recovery-without-bundle.md) prepares public pinned assets and
a separately versioned corpus without the lost archive. On the 24 GiB Air, the
loader synchronizes and releases unused MPS cache after each original Metal
conversion, restoring the scoped converter afterward. Arithmetic, FP16 runtime,
4-bit/group-64 and context/output budgets are unchanged. Generation rejects NaN,
positive infinity and fully masked score rows without repairing scores; ordinary
negative-infinity token masks remain allowed. Unused cache is released after
each call, including errors. One Qwen instance is loaded after BGE is released.
See [measurements and limits](experiments/air-recovery.json); successful loading
or pipeline completion alone does not imply a valid draft.

The recovery CLI p01 request completed with one accepted partial-draft claim
in 188.75 seconds including process startup/imports; its two visible citations
matched exact sources. The p06 request completed but withheld an overlong draft.
The separate ten-question recovered baseline yielded nine visible drafts and
one output-limit failure; both controls abstained. These are single developmental
runs, not an independent answer-quality evaluation.

The 9B comparison retains baseline review v4. [Historical semantic-review measurements](day21-semantic-reliability.md)
found material support errors in both that v4 prompt and an experimental seven-axis
v5 reviewer; v5 regressed on controls and remains an explicit legacy research
selection in `generate_answers.py` or `generate_answer`. The default 4B uses the
named axes-first v4 prompt; it rejects an explicit v5 request. The user's separate
[manual assessment](qwen4b-manual-human-assessment.md) found that both named model
reviewers failed the zero-known-bad-acceptance gate. None of these reviewers
provides independent clinical validation.
