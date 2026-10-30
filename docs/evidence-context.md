# Evidence context and source mapping

MedLens can now pack exact abstract excerpts into a bounded context for the
[experimental answer generator](cited-answers.md). This stage selects source text and assigns citation
IDs. It does not generate answers, determine whether a claim follows from a
quote, or establish clinical validity.

## Selection policy

The default pilot run uses the saved dense + Qwen article ordering from the
active `pilot-qrels-v2` profile. No reranking is repeated. BGE encodes the ten
questions again on MPS to score all existing windows within each article;
query-vector hashes are recorded locally. Stored passage vectors are reused.

The initial limits from the local-inference ADR apply:

- Up to six fragments, up to two per article.
- At most 4000 Qwen tokens for the exact serialized evidence payload, including
  citation IDs, PMIDs, titles, journal, available journal-issue publication date,
  section labels and excerpt text.
- Questions, instructions, chat-template overhead and answer tokens are outside
  this evidence budget. Generation costs are measured separately in the
  [answer pilot](cited-answers.md).

Articles retain their input order. The first pass offers one excerpt per article;
a second pass offers another only if capacity remains. Each pass skips excerpts
that cannot fit, without cutting or rewriting their text. The selector continues
to later windows/articles, and can return fewer than six or none. Source ID
assignment is deterministic (`E1`, `E2`, ...), scoped to one context.

For these treatment-efficacy questions, windows from Results/Conclusions have
priority within a paper, followed by unstructured sections, then other sections.
NLM categories and a small explicit set of English labels establish that order;
cosine similarity breaks ties within each category and source order breaks exact
score ties. `ContextConfig(prefer_outcomes=False)` removes this structural
preference for other experiments. It is not a general rule for questions about
methods or mechanisms.

Clearly labeled registration and funding sections are excluded. The list is
explicit rather than a classifier; unlabelled administrative text can still slip
through. Negative cosine scores are retained without inventing a relevance
threshold. Title-only records provide no abstract evidence and are skipped.

Within one article, selected fragments cannot overlap in the same source
section. This removes shared overlap text from adjacent windows; it does not
perform semantic deduplication or deduplicate separate publications of one trial.
With ten usable articles and the six-fragment limit, the first pass usually
fills the context from six distinct articles.

## Citations and provenance

Every fragment retains its PMID, URL, dataset/text version, window ID, original
section path, exact character span and snapshot provenance outside the compact
model payload. Source text comes from the stored character slice, never from
decoding BGE input IDs. Labels, human/AI relevance grades, rank scores and window
similarities are not included in the evidence payload.

`locate_quote(context, "E1", quote)` accepts only nonempty verbatim text contained
in that selected fragment. It rejects unknown IDs, case/whitespace changes,
unmatched quotes and altered context source mappings/payloads. Repeated matches
are returned with an ambiguity flag; their offsets are Unicode code-point offsets
in the parsed abstract section. Quote matching verifies source location, not
entailment. The answer generator preserves this source context and resolves
references against its trusted map. It renders the excerpts as a sentence catalog
with IDs such as `E1.Q1`, counts that representation against the same 4000-token
budget, and rejects an over-budget catalog. This is a separate payload format;
the original context payload and identity remain unchanged.

Empty contexts return `no-source-excerpts`. A nonempty context is only
`source-excerpts-selected`: it is not a declaration that the question has
sufficient or relevant evidence. The generation stage must still assess
population fit, scope, unsupported conclusions, contradictions and abstention.

## Pilot check

The first cosine-only selection exposed a failure: one source excerpt was just
a PROSPERO registration number, and twelve were Background sections. Both were
traceable source text, demonstrating that provenance checks do not establish
answer usefulness. That [initial aggregate run](experiments/evidence-context.json)
is retained with its original implementation hash.

After the section policy change, the [current run](experiments/evidence-context-outcomes.json)
contains 60 excerpts from six distinct papers per question. Of those excerpts,
36 are labeled Results/Conclusions, 21 are unstructured, one is Background and
two are Purpose of Review. No selected fragment is a labeled registration or
funding section. This describes the source mix, not an answer-quality score.

| Question | Fragments | Qwen evidence tokens |
| --- | ---: | ---: |
| p01 | 6 | 1119 |
| p02 | 6 | 1177 |
| p03 | 6 | 1288 |
| p04 | 6 | 1455 |
| p05 | 6 | 1006 |
| p06 | 6 | 1021 |
| p07 | 6 | 1481 |
| p08 | 6 | 1256 |
| p09 | 6 | 1785 |
| p10 | 6 | 700 |

All 60 excerpts matched their original dataset spans and passed full-excerpt
quote location checks. The evidence payloads use 700–1785 tokens, below 4000;
synthetic tests exercise actual budget rejection, overlap, source corruption,
title-only/empty evidence, exact and ambiguous quote matches, Unicode offsets,
negative scores, and the service-section failure. All 124 offline tests pass.

Selection plus artifact validation took 0.118 seconds over ten questions,
excluding encoder/tokenizer loading, query encoding and output writing. Batched
query encoding took 0.114 seconds. Qwen's tokenizer is used, but its 9B model is
not loaded. These timings do not measure answer generation.

## Run

Requires the existing parsed/window/vector artifacts and cached pinned Qwen
tokenizer. All source labels and full excerpt payloads remain local.

```sh
HF_HUB_OFFLINE=1 python scripts/prepare_evidence.py \
  --output results/evidence/pilot-v1-dense-qwen-004.json \
  --summary docs/experiments/evidence-context-outcomes.json
```

Choose fresh paths for subsequent runs. `--method dense`, `rrf`, or `rrf_qwen`
selects another frozen article ranking. The default profile pins all historical
inputs; replay verifies their datasets, rankings and outputs before selecting
text. The full local report includes exact payloads and citation maps; the public
summary includes configuration/provenance hashes, window IDs, counts and timings.
