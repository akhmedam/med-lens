# Cited answers from abstract excerpts

MedLens can generate an English research answer from a saved evidence context.
Each displayed draft finding carries one or more source IDs, verbatim quotations and
source offsets. A second call to the same local Qwen instance reviews claim
support before the answer is shown. This is an experimental generation stage;
model review is not independent expert verification or a grounding guarantee.

## Protocol

The pinned Qwen3.5-9B model and 4-bit MPS runtime are shared with reranking.
`QwenRuntime.generate` keeps the rerank default of 128 output tokens. Answer
drafts allow 768, support review 512; both use greedy decoding with thinking
disabled. Per-call limits and input/output counts are recorded separately from
runtime defaults. One model instance handles all calls sequentially.

The generator receives the question and a quote-indexed rendering of the source
excerpts prepared on Day 12. Metadata and source text remain bound to the original
context; each sentence receives an ID such as `E1.Q1`. The transparent sentence
boundary heuristic may split abbreviations, but every catalog entry remains an
exact contiguous source slice. The 275 entries in the pilot were checked against
the source map. Repeated identical sentences are distinguished by ID and offset.
The generator receives no relevance labels. Source text, question and draft are
treated as untrusted data. The evidence budget remains 4000 tokens;
quote IDs and sentence metadata are counted in that budget. The full prompt limit
is 6500 input tokens including instructions/chat overhead.
Neither evidence nor overlong prompts are silently truncated.

The draft is JSON with a status, bounded cited claims and controlled
limitation codes. The prompt requests one to three claims; the parser accepts up to four.
Every claim needs one to three exact catalog quotations. Unknown IDs, changed
source spans, duplicate JSON keys, invented citation fields,
model-written PMID/URL references and malformed or truncated output are rejected.
Limitations may appear at the root or inside claims; both layouts are explicitly
recognized and combined without changing claim text or references. Unknown fields
and limitation codes remain errors. PMID links come from the trusted source map.
The renderer uses accepted claims only; it never displays an unchecked free-text
summary alongside them.

Support review returns a verdict and three explicit axes for every draft claim:
population, comparison and outcome, plus a controlled reason code. Free-form
review explanations are rejected to keep the 512-token review bounded. Code
accepts a claim only when the verdict
is supported, population/outcome match and comparison matches or was not requested.
Mismatches and unclear scope withhold the claim even if the model's overall
verdict says supported. The reviewer sees only the cited excerpts and metadata.
It can still make a confident mistake on all axes; passing these checks does
not prove scientific correctness.

Displayed text is explicitly an unverified research draft requiring independent
review, even when all model-support axes pass. Completeness is reported separately
as answered, partial, or insufficiently supported. Partial answers retain only
accepted claims. Empty contexts skip
inference. A schema/quote/output-limit failure is `validation_failed`, with no
visible draft; it is distinct from a finding that the supplied excerpts do not
support an answer. No automatic repair or retry changes the model output.

## Pilot and limitations

The first coarse support-review run was stopped after p01. It accepted a
teenager-only finding in an adult answer, and accepted a comparison between
monotherapies derived from each monotherapy versus combined treatment. This led
to the explicit population/comparison/outcome protocol and non-equivalence
instructions. The stopped report remains local; it is not counted as a completed
quality evaluation. Protocol changes used development cases, so the subsequent
run is not held out.

The final pilot uses the frozen Day 12 dense + Qwen contexts under qrels v2.
The [free-quote protocol run](experiments/cited-answers-free-quotes.json) then
withheld nine of ten pilot drafts on validation failures. Observed failures
included schema-layout differences and non-verbatim quotations. It also exposed
a loss of the "more severely depressed" subgroup qualifier in a claim that the
reviewer marked supported. Choosing quote IDs fixes quote copying, not these
semantic problems. The quote-ID run still accepted that missing subgroup
qualifier on p01 and
accepted a combined-treatment comparison as a match for a monotherapy question.
Its reviewer rejected the adolescent claim but did not reliably enforce all axes.
The final renderer therefore presents model-reviewed findings
as unverified drafts and shows their exact quotations for inspection.

The quote-ID run completed all ten pilot questions. Four yielded visible drafts
(one `answered`, three `partial`) with nine accepted claims; six were withheld
as `validation_failed`. Failures were three excessive claim counts, one excessive
citation count, one invalid review-axis value and one malformed JSON draft.
These are format outcomes, not answer-accuracy scores. No schema or token-limit
failure is converted into an answer.

Source inspection by Codex also found an omitted older-adult qualifier and an
SSRI-specific claim inferred from an antidepressant-class result. This inspection
is an AI assessment of the displayed excerpts, not independent expert validation;
no clinical correctness rate is reported.

Two additional controls supply an empty context and a real adolescent-only
trial for the adult comparison question. Control outcomes and the actual
model-support decisions are recorded in the [aggregate report](experiments/cited-answers.json).
Both controls returned `insufficient_evidence`: empty evidence made no model
call, and the adolescent-only context caused the generator to abstain.

The evidence catalog used 832–2287 Qwen tokens. Median summed model-call time
per pilot question was 48.80 seconds (range 36.35–65.53), excluding model load,
tokenization and report writing. The p01 draft was reused from the interrupted
quote-ID run; its review was fresh. These component timings include that original
draft timing, so they are not fresh end-to-end measurements. Model loading took
25.48 seconds. Process peak RSS was 13.19 GiB across load and inference; it is not
a separate GPU memory measurement or an additive device peak.

An offline replay reproduced all twelve statuses, claims and review decisions
using seventeen cached calls and no new inference. All source/code hashes,
279 catalog entries including controls, and twelve displayed citations passed
source-span checks. The ten pilot contexts contain 275 distinct catalog entries;
the control adds four reused source sentences. See the
[validation and source-inspection record](experiments/cited-answers-audit.json).
Full prompts, raw drafts/reviews, exact quotations and source maps stay local.

## Run

Requires the saved context report, its pinned local datasets/windows and cached
model/tokenizer. The kernel resolver may require network access for the normal
publisher/version check, while weights load with `local_files_only=True`.

```sh
python scripts/generate_answers.py results/evidence/pilot-v1-dense-qwen-004.json \
  --include-controls \
  --output results/generation/pilot-v1-cited-004.json \
  --summary docs/experiments/cited-answers.json
```

Choose fresh output paths on repeated runs. `--reuse-outputs <local-report>`
reuses a model call only when its prompt hash and output-token limit match; the
context/profile and model revision must also match. Reused calls carry their
source report hash. Parsing or gate changes can be evaluated without regenerating
identical outputs. Prompt changes (including the quote-indexed payload) require
new inference. Per-stage component timings include the original measurement for
a reused call and are not a new end-to-end latency observation. Before inference,
the runner checks
the active evaluation/profile hashes, query order/text, article ranking,
context identity, selected source windows and Qwen tokenizer fingerprint. Each
completed case is checkpointed; interrupted/failed runs retain their status.
The local JSON includes the rendered answer, citations, raw model calls and
review decisions. The public summary excludes individual labels and article text.

This command consumes a prepared context; a live search-to-answer CLI/API and
reuse of the same instance across a live rerank-to-generation request remain
integration work. Broader answer evaluation, independent human checking,
contradiction/abstention coverage and testing near the full 4000-token evidence
budget remain necessary before treating this as a final answering system.
