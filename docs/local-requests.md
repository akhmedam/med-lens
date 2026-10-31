# Local search-to-answer requests

`scripts/ask.py` runs a fresh English question against one verified local
collection. It saves a human-readable `report.html` and a full `run.json` in a
new directory. No pilot relevance labels or saved rankings enter this path.

## Run

From the project root, with the existing virtual environment and cached models:

```sh
.venv/bin/python scripts/ask.py data/parsed/pilot-v1/p06 \
  --windows data/windows/pilot-v1-002/p06 \
  --embeddings data/embeddings/pilot-v1/p06 \
  --query "Does cognitive behavioral therapy reduce anxiety symptoms in adults with generalized anxiety disorder?" \
  --output-dir results/requests/gad-example
open results/requests/gad-example/report.html
```

The default mode returns dense search results and selected source passages.
`--generate` enables experimental answer drafts; `--rerank` enables optional
Qwen reranking of the dense results. These flags work independently. `--top-k`
accepts one to ten articles and defaults to ten. `--device cpu` changes BGE only;
Qwen generation and reranking require the recorded MPS runtime.

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
   strict gates from the Day 13 baseline are unchanged.

The default mode loads BGE and the cached Qwen tokenizer, but no Qwen weights.
Retrieval/tokenizer-only operation can run with `HF_HUB_OFFLINE=1`. With Qwen,
the normal Metal-kernel publisher/version check may need network access even
though model weights load from the local cache.

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

## Validation

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

A pooled corpus, new PubMed candidate fetching, a served API, independent answer
assessment and more reliable generation remain open work.
