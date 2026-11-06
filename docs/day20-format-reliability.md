# Day 20: generation format reliability

Status: implementation and offline checks prepared on 2026-11-06. No new model
inference or before/after pilot has run. Improved Qwen compliance is **not yet
demonstrated**. The accepted Dense → evidence → Qwen3.5-9B pipeline is unchanged.

## Changes

- The v4 draft prompt includes a JSON Schema with the current quote-ID enum,
  root limitation codes, status/claim-count consistency and bounded citations.
  The review schema enumerates the actual claim IDs and all permitted verdicts,
  axes and reason codes. These are instructions to the model, not constrained
  decoding or a guarantee that the output is valid.
- The parser now enforces the prompt's three-claim limit; Day 13 accepted four.
  Each claim still requires one to three distinct exact catalog quotations.
  Legacy free-quote parsing and per-claim limitation layouts remain supported;
  the production prompt requests quote IDs and root limitations only.
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
source spans and held-out reservation are unchanged. The larger schema prompts
must be measured with the real tokenizer near the full evidence budget; overflow
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
The raw contexts and calls were unavailable during this work. No claim is made
that the old failures would pass the new prompt. The tighter cap may withhold
historical four-claim drafts; use the Day 19 code when replaying the old protocol.

## After recovering the original artifacts

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
