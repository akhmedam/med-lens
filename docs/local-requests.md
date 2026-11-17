# Offline research and replay requests

The supported application is the live [MVP CLI/API](mvp.md). This page documents
`scripts/ask.py`, a research wrapper for verified local dataset/window/embedding
artifacts. It does not search PubMed or expand its supplied candidate set.
Existing saved source collections remain useful for fixed-source experiments;
they are not an alternative freshness policy in the public application.

## Run

From the prepared checkout, choose a new output directory:

```sh
.venv/bin/python scripts/ask.py data/parsed/recovery-20261106/p06 \
  --windows data/windows/recovery-20261106/p06 \
  --embeddings data/embeddings/recovery-20261106/p06 \
  --query 'Does cognitive behavioral therapy reduce anxiety symptoms in adults with generalized anxiety disorder?' \
  --generate --answer-style source-findings --output-dir results/requests/local-findings-NEW
```

The source-findings format selects exact study-context/finding quotations with
applicability cautions. It does not generate a factual synthesis or run the
prose-support reviewer. Nonempty findings remain `partial`, and semantic support
and independent validation remain unverified.

To study the distinct prose draft/review path, pass `--generate --answer-style
draft` explicitly. Its 4B compact policy uses `cited-answer-v8` and six named
`claim-support-v4` fields under `axes-first-v2`. Exact citations and logical
review consistency do not establish factual support. Historical explicit 9B
comparisons retain their legacy policy. These choices are research options, not
public CLI/API alternatives.

Without `--generate`, this wrapper runs diagnostic search/excerpt selection.
`--rerank` independently enables research Qwen reranking; it is absent from the
MVP. `--top-k` accepts 1–10 and defaults to ten. `--device cpu` changes BGE only.
`--answer-model 9b` and `--answer-model 9b --engine transformers` select explicit
historical comparisons. They are not automatic fallbacks and are not required
for native 4B setup.

Search covers only the supplied dataset. A new question does not acquire new
papers. Similarity scores are not relevance probabilities. The artifact loader
verifies actual source versions, ordered mappings, normalized vectors and exact
spans before packing evidence. Saved catalogs keep their recorded identities;
older catalog boundaries must not be relabeled as the complete-source-sentence
policy without rebuilding and verifying the context.

## Lifecycle and results

The wrapper validates artifacts and encodes the question once, ranks articles
by maximum-window cosine similarity, and releases BGE before Qwen. Exact
source-findings and research prose modes have distinct contexts/contracts.
A Qwen instance is shared only across requested model stages within that job.
Empty evidence needs no model call. Invalid output is withheld; completed search
and evidence remain available after a model-stage failure.

The wrapper saves private diagnostic `report.html` and `run.json`, checkpointed
with atomic replacement. Their raw model/diagnostic contents are different from
the public CLI/API projection. Do not substitute those files for the public
export intended for sharing. Existing output directories are refused. An abrupt
kill can leave a running checkpoint; it is not a completed execution.

Execution and answer status are separate. Research-wrapper exit codes are:

| Code | Meaning |
| --- | --- |
| 0 | Completed execution, including explicit insufficient evidence |
| 1 | Setup, artifact, source, model or inference failure |
| 2 | Invalid arguments or an answer-validation failure; inspect the saved report |
| 130 | Interrupted execution with prior checkpoints retained |

The shared corpus and licensed PMC subset remain research artifacts. The
[historical observations](#historical-9b-validation) below retain their original
models, source pools and measured costs; they are not final MVP validation.

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
answer support receive independent assessment. A served API was open work at
that time; the current [local API](local-api.md) is implemented separately.

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

The 9B comparison retains baseline review v4. [Historical semantic-review measurements](semantic-support-diagnostics.md)
found material support errors in both that v4 prompt and an experimental seven-axis
v5 reviewer; v5 regressed on controls and remains an explicit legacy research
selection in `generate_answers.py` or `generate_answer`. The default 4B uses the
named axes-first v4 prompt; it rejects an explicit v5 request. The user's separate
[manual assessment](qwen4b-manual-human-assessment.md) found that both named model
reviewers failed the zero-known-bad-acceptance gate. None of these reviewers
provides independent clinical validation.
