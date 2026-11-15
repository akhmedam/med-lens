# Prospective MVP operational confirmation

This addendum is declared on 2026-11-15 before opening the twelve held-out
question texts or their model outputs. The candidate PubMed expressions were
inspected beforehand; that limited exposure is disclosed in the frozen manifest.
It is not a claim of fully blind test design. Model/prompt/selector tuning must
finish on development evidence before the freeze.

## One selected application

The application uses bounded live PubMed acquisition, BGE-base-en-v1.5 on MPS,
Dense top-ten article ranking, exact source evidence and the existing V2 scoped
source-findings selector with Qwen3.5-4B MLX four-bit. The selector, scope guards,
source-sentence policy, generation limits and model revisions are unchanged.
The answer style is source-findings only; generation uses no scientific-support
certification and nonempty findings remain partial. The candidate limit is 100.
Question text is preserved and passed through the current recorded lexical
planner without an explicit PubMed override. No per-question alternative mode,
prompt, candidate limit or retry policy is selected after inspecting results.

The original `benchmarks/heldout-v1/queries.jsonl` and `candidate-plan.json` stay
unchanged. Their SHA256 identities are respectively
`7d1fa94cbf999bca44a139f60a3646e505565244712a6b60291ccae311a7035f` and
`45d3680c16ffc9c595817243b65099d5fcd5a5918881d5b86593dce45dfa138e`.
The original 30-candidate, four-ranker comparison is retained as a historical
protocol; this separately declared application check does not pretend to fulfill
that comparison. It uses the twelve frozen question texts in their original
order, with fresh sources rather than the absent reserved dataset directories.

## Freeze before reading questions

```sh
.venv/bin/python scripts/validate_mvp.py freeze results/final-validation-20261115
# Review manifest.json and its identity before its one execution.
.venv/bin/python scripts/validate_mvp.py run results/final-validation-20261115
```

Freeze hashes opaque question bytes without deserializing the question rows. It
records all application Python source hashes, this harness/protocol, the central
application policy snapshot/ID, active dependency locks and asset declarations,
package versions, the native worker environment, model/tokenizer revisions, actual asset
file checksums and the selector system-prompt hash. `manifest.json` and its
directory are created exclusively. The source, policy, assets, environments and
question identities are rechecked before execution.
Source identity is checked again before every case and at final acceptance;
each persisted job must carry the same central application policy. Actual model
and tokenizer file hashes/sizes must match the declared asset checksums, including
model weights, rather than merely recording a changed file as a new identity.

Run creates an exclusive `execution-started.json` marker before parsing question
texts. A second run, resume, force overwrite, per-case retry or selection of a
better output is not supported. Existing bounded PubMed transport backoff stays
part of the unchanged acquisition policy. A process failure leaves the original
attempt and its partial artifacts intact; it is not quietly replaced.
Attempted questions, accepted job IDs and completed case records are journaled
separately, so interruption cannot erase an already accepted attempt from counts.

## Operational checks

One JobService processes the twelve questions sequentially. Each request keeps
its private raw snapshot, dataset/windows/vectors/run report and public job
envelope. Each case checks completed fresh acquisition, public/private projection
identity, complete context/source bindings, citation text versions and exact
source offsets, a closed runtime when loaded, and reaping of native worker PIDs
when observed. A partial or insufficient-evidence answer can be operationally
successful; an error, invalid answer, source-binding failure, uncertain cleanup
or an observed live native worker fails that case. Uncertain service health stops
the remaining cases rather than risking another native computation.

Current parent RSS and descendant RSS are sampled separately every 0.25 seconds;
native MLX workers are distinguished from Python helpers. Observed RSS values,
existing MPS stage samples and native Metal/RSS peaks overlap and must not be
added. Parent high-water is not used as evidence of a leak. Native PID observation
can be absent for a short job and is then explicitly unknown. This check claims
neither whole-machine/swap peaks nor long-run leak freedom. Development repeats
and cold/warm latency measurements are a separate declared check, with actual
passage-cache hits disclosed. No latency SLA or threshold selected after results
is introduced.

The existing deterministic tests supply malformed/unsupported-format input,
empty evidence, upstream/model errors, stale-fallback exclusion, busy admission,
cleanup/storage failure, control-character rendering and repeated-signal checks.
Their synthetic origin remains separate from fresh-source operational evidence.
Language/topic scope is governed by the final application input contract, not
inferred from an empty PubMed response.

## Reference and result interpretation

New held-out candidates have no finalized retrieval reference and generated
outputs have no new independent human claim/answer reference. Historical human
labels and assistant development judgments are not transferred to these sources.
Retrieval nDCG/Recall/MRR and independent answer-quality estimates remain null.
Unknown or absent judgments never become irrelevant or supported labels.

The summary reports attempted/expected cases, operational pass/fail, source
integrity, answer outcomes, latency/memory scopes and missing reference coverage.
Passing execution does not satisfy independent quality/release conditions.
The independent quality gate stays `not-established` and
`release_quality_conditions_satisfied=false`. Any later assistant source-bound
review is separate AI-origin diagnostic evidence, not a human reference or proof
of general biomedical or clinical reliability. After observing these test
outputs, a behavior change requires a future untouched set for confirmation.
