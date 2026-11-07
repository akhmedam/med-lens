# Day 20: generation format reliability

Status: offline implementation was prepared on 2026-11-06; runtime recovery and
a new-corpus developmental baseline followed on 2026-11-07. The historical raw
artifacts remain lost, so there is no controlled before/after comparison with
Day 13. The accepted Dense → evidence → Qwen3.5-9B pipeline is unchanged.
See [Air recovery](experiments/air-recovery.json) and the
[repeatable recovery procedure](recovery-without-bundle.md).

## Changes

- The v6 draft prompt includes a JSON Schema with the current quote-ID enum,
  root limitation codes, status/claim-count consistency and bounded citations.
  The review schema enumerates the actual claim IDs and all permitted verdicts,
  axes and reason codes. These are instructions to the model, not constrained
  decoding or a guarantee that the output is valid.
- The parser now enforces the prompt's three-claim limit; Day 13 accepted four.
  Each claim still requires one to three distinct exact catalog quotations.
  The current wire format requires exact quote-ID strings and root limitations.
  The separate legacy object mode retains free-quote/per-claim layouts; no output
  is coerced between formats. Developmental v4/v5 probes repeatedly returned ID
  strings against object contracts, motivating this explicit protocol change.
- Format failures record a stage and stable error code, distinguishing claim and
  citation limits, JSON errors, review coverage and enum violations. Invalid
  output remains withheld in full; no repair, retry or array trimming occurs.
- Both generation and request identities include the contract implementation.
  Changed prompts invalidate old prompt-hash caches. Aggregate generation reports
  now preserve prompt versions and validation diagnostics.
- `compare_answer_formats.py` counts all completed pilot cases, separates controls
  and distinguishes format failures from insufficient-evidence outcomes. It pairs
  outcomes only when context/profile/label identities, cases, model revision,
  quantization, package versions and observed output budgets match. Otherwise it
  gives descriptive counts with no paired delta. Reports are measurements of
  visible format outcomes, not clinical accuracy or independent support review.

Evidence/input budgets remain 4000/6500 tokens. Draft/review limits remain 768/512
tokens. Model revision, Metal 4-bit/group-64 runtime, semantic support gates,
source spans and held-out reservation are unchanged. Real-tokenizer packing
passed with 4000 quote-indexed evidence tokens and
4774 draft / 5015 review input tokens on a synthetic source case using the
production prompt builder and fake model outputs. This is
not inference at full budget or proof for all sentence distributions; overflow
still fails without truncation. Same-model review can still miss semantic errors.

## Checks available without the missing bundle

Run the full offline suite and dependency check:

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pip check
```

The [offline check record](experiments/day20-format-offline.json) reports **218 passing
tests** and a successful `pip check` on the Air.

The tests exercise request-specific contracts, unchanged token budgets, the new
claim cap, malformed JSON/review fields, exact-source citation rejection, semantic
axis gating, cache identity, comparison denominators and incompatible reports.
They use synthetic model outputs; passing them is not an inference benchmark.

The saved Day 13 aggregate still reports four visible drafts, six format failures
and nine displayed claims across ten pilot questions. Its two controls are separate.
The historical raw contexts and calls remain unavailable. No claim is made
that the old failures would pass the new prompt. The tighter cap may withhold
historical four-claim drafts; use the Day 19 code when replaying the old protocol.

## Historical replay route (requires the lost artifacts)

1. Verify and restore the bundle using a Day 19-compatible checkout, following
   [the handoff guide](handoff.md). Code identity mismatches on this feature branch
   are expected; do not alter the transfer manifest to bypass them. Keep recovered
   pilot-v1 data separate from the newly fetched recovery-20261106 collections.
2. Complete the staged Air checks, including short Qwen inference and a full
   Dense request with `--generate`, without `--rerank`. Record memory and errors.
3. Measure v3 and v4 on identical pilot contexts, using fresh result paths. For a
   comparison with the original Day 13 result, reuse its original Dense + Qwen
   contexts; they differ from the accepted primary Dense-only ranking. Measure
   the primary pipeline separately, with its own unchanged before/after contexts.
4. Compare the completed aggregate reports:

```sh
.venv/bin/python scripts/compare_answer_formats.py \
  docs/experiments/cited-answers.json \
  results/generation/day20-summary.json \
  --output results/generation/day20-format-comparison.json
```

Use a fresh output path. A different context-report hash prevents a controlled
comparison, even when question IDs match. The tool cannot independently authenticate
an aggregate report; retain raw calls, exact prompts, source checks and provenance.
Inspect fewer visible findings as well as malformed outputs: a stricter cap is a
protocol change, not evidence of improved semantics. If raw data cannot be recovered,
establish a new paired baseline on the rebuilt corpus instead of substituting it
into the frozen profile. Do not use held-out questions for tuning.

New recovered-corpus reports explicitly omit historical relevance judgments.
Two rebuilt reports may be paired only with identical contexts, profiles and an
explicit matching unjudged policy; mixed scopes remain incomparable. These are
format comparisons, never retrieval-quality or clinical-correctness estimates.


## Completed rebuilt-corpus baseline

The [new aggregate](experiments/recovery-generation.json) records six answered,
three partial and one output-limit failure across ten development questions;
22 draft claims survived model support review. Both controls abstained. The
[historical comparison](experiments/recovery-generation-comparison.json) reports
no paired delta because contexts/profile/judgment policy differ. The
[Air audit](experiments/air-recovery.json) records 247 passing unit tests, exact
source checks and offline replay of all twelve cases using twenty cached calls.
This establishes a new baseline, not independent semantic validation. Long-output
compliance, a controlled prompt comparison on fixed new contexts and Day 21
semantic checks remain open.
