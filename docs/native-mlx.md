# Native MLX inference on Apple Silicon

The live and local request CLIs default to the pinned **Qwen3.5-4B community MLX
artifact**, selected by the user for development. They produce research drafts
that require human review. The [model decision](adr/0005-qwen4b-development-default.md)
does not assert clinical reliability or equivalence with 9B. Both model reviewers
failed the zero-known-bad-acceptance gate in the manually annotated pilot.

MLX runs in an isolated Python subprocess. BGE and the pinned official answer
tokenizer remain in the main environment. Qwen9B is an explicit comparison;
missing dependencies, assets or a failed backend never trigger an automatic
model substitution.

## Prepare the default 4B profile

From the checkout, after creating the main Python 3.11 environment and installing
`requirements-macos-arm64-day19.lock`:

```sh
.venv/bin/python -m venv .cache/venvs/mlx
.cache/venvs/mlx/bin/python -m pip install -r requirements-mlx-macos-arm64.lock
.venv/bin/python scripts/prepare_mlx.py --download
```

This prepares 4B directly. **Original 9B weights are not required.** The explicit
`--download` flag permits fetching absent pinned public files; without it,
preparation uses verified local assets and fails if files are missing. Existing
files must match their pinned size and SHA256. Corrupt files are not silently
overwritten. A completed model output must be fresh; failed staging remains
available for inspection.

Preparation verifies the community model config/weights, safetensors layout,
BF16/FP32 parameter policy and tied quantized embedding/output head. It copies
the checked model without dtype conversion and writes `medlens-source.json` only
for a completed artifact. It also prepares three separately pinned official
tokenizer files and `tokenizer-source.json`. Preparation needs no GPU or native
MLX import; inference requires an Apple GPU and the isolated environment.

| Profile | Model and weight revision | Prepared directory | Floating remainder |
| --- | --- | --- | --- |
| Default `4b` | `mlx-community/Qwen3.5-4B-MLX-4bit` at `32f3e8ecf65426fc3306969496342d504bfa13f3` | `.cache/models/qwen3.5-4b-mlx-4bit-g64` | BF16, with `A_log` FP32 |
| Comparison `9b` | `Qwen/Qwen3.5-9B` at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `.cache/models/qwen3.5-9b-mlx-linear-4bit-g64` | FP16 embedding/head and remainder, with `A_log` FP32 |

Both use affine 4-bit/group-64 weights. The 4B profile includes the tied
embedding/head in quantization. The selected 9B comparison quantizes linear
layers except the output head. These are different models and arithmetic policies.

The 4B community weight file is 3,034,300,695 bytes, SHA256
`5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db`.
Its card names Qwen3.5-4B as the source but does not disclose an exact upstream
official revision or converter commit. Those provenance fields remain unknown.
The official **tokenizer** is pinned independently to `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; that does not establish the upstream
revision of the community weights.

For an existing local 4B payload, `--source-dir` and `--tokenizer-source` select
the files to verify. They do not bypass pins. Do not relabel existing weights by
editing their manifest.

## Optional 9B comparison

Restore the original pinned 9B snapshot through the existing model-asset workflow,
then convert it offline with the same isolated environment:

```sh
.venv/bin/python scripts/prepare_mlx.py --model 9b
```

This is the earlier local 9B conversion. It hashes original source files,
converts with MLX-LM 0.32.2 and validates prepared files. It does not download
9B weights. `--coverage` and a fresh `--output` can prepare another quantization
policy as an explicit experiment; changing coverage is not an equivalence claim.

## Run and inspect

```sh
.venv/bin/python scripts/ask_live.py \
  --query "Does exercise reduce depressive symptoms in adults with depression?" \
  --limit 100 --output-dir results/requests/native-4b-example
```

For explicit comparisons, add `--answer-model 9b` for native 9B, or both
`--answer-model 9b --engine transformers` for the original MPS/Metal backend.
Transformers is a 9B comparison and rejects the 4B profile. `--mlx-model-dir`
selects another verified native artifact for the declared profile;
`--mlx-python` selects the isolated interpreter.

Local replay uses `scripts/ask.py --generate` with dataset/window/embedding
directories. Primary requests omit the optional reranker. Search-only and
empty-source requests do not load Qwen weights; nonempty evidence packing still
uses the pinned answer tokenizer. Model failures preserve completed retrieval
and source evidence. Every live request searches/fetches PubMed anew.

The request records the selected model, backend, answer policy, tokenizer source,
manifest/artifact identity, precision and package versions. Actual input IDs are
hashed separately from source messages. Weights, source texts and raw outputs
remain local. `scripts/generate_answers.py` remains a frozen **9B experiment**
entry point; it does not inherit the live/local CLI's new default.

## Answer and runtime contracts

The 4B `compact-v1` profile uses `cited-answer-v7`: a short draft instruction and
the complete unchanged quote catalog. Review uses the six-field v4 taxonomy
with the named `axes-first-v1` prompt. Population, comparison and outcome precede
the explicit verdict and reason. The decoder and strict post-parser reject
inconsistent combinations. Passing axes alone never implies factual support;
`not_supported/overstatement` remains possible with passing axes.

These are named JSON objects, not positional arrays. Draft constraints receive
the original schema. The native worker explicitly lowers unsupported schema
features and records the source/compiled schemas and remaining strict-parser
checks. Claim count, quote IDs and status/cardinality branches remain bounded.
There is no truncation, post-hoc repair or inferred review default. Constraints
can guarantee permitted combinations of fields; they cannot guarantee truth.

The 9B comparison retains `legacy-v6` prompts and its legacy review path. An
explicit v5 semantic-review request is unsupported by `compact-v1` and fails
clearly; it is not silently reduced to v4. Historical v5 experiments remain
available through the legacy route.

Both profiles preserve the 4,000-token evidence budget, 6,500-token full answer
input limit, 768 draft-token cap, 512 review-token cap, maximum three claims and
one to three exact citations per claim. A capped output is withheld even when
its last token is EOS. Both backends stop on source end-of-text 248044 or chat
end 248046; terminal EOS is counted. The earlier Transformers implementation
inherited only source EOS, so old measurements can include a stopping-policy
change as well as a runtime change.

One worker loads weights once for draft plus review. Each call gets fresh,
unquantized KV/recurrent state. Grammar-cache reuse is not answer/state replay.
Compact answer-call reuse requires matching model/revision/artifact/policy and
decoder contract, so a saved 9B response cannot satisfy a 4B call. A bounded
JSON-lines protocol validates IDs, caps and response/session identity. Success,
error, interruption and timeout close/reap the worker; no CPU/model fallback is
introduced. Stderr stays separate from the protocol.

Raw NaN, positive infinity and fully masked score rows fail before sampling.
Allowed negative-infinity masks remain allowed. Startup checks the manifest,
file sizes and config/tokenizer hashes. Preparation hashes complete files;
`MlxRuntime(verify_weights=True)` can repeat full weight verification without
hashing gigabytes on every request.

## Limited manual evidence

The user independently filled and finalized a blind 30-claim reference and six
answer pairs. These cover 32 claim occurrences, six related source pools and
**two independent development questions**, with one human annotator. The
assistant validated bindings and counted the supplied labels; it did not supply
or revise semantic labels. The assessment did not reuse relevance qrels or
held-out questions.

Of 24 human-eligible unique claims, the named 4B reviewer retained 16 and 9B
retained 15. However, 4B accepted 3 of 5 human comparison-mismatched claims and
one human-uncertain claim; 9B accepted 4 of those 5 mismatches. **Both failed the
zero-known-bad-acceptance gate.** These mismatches concern the requested contrast;
they must not be described as five factually fabricated claims.

The user rated 4B answers usable in 5/6 contexts and complete in 2/6, compared
with 9B in 4/6 and 0/6. The 4B workflow still failed the predeclared all-context
usability gate. Its paired completeness check found no loss against an adequate
9B answer, but no 9B answer was rated adequate, so that is not a certificate of
completeness. An explicit rating on an empty answer is preserved without
converting it into an asserted false fact.

The user selected 4B for faster development iteration despite these limits.
Same-model review remains a heuristic, and displayed answers remain research
drafts requiring human review. Neither model is a human reference or an approved
autonomous biomedical judge. See the [human assessment](qwen4b-manual-human-assessment.md),
its [aggregate record](experiments/qwen4b-manual-human-assessment.json) and
[ADR 0005](adr/0005-qwen4b-development-default.md).

## Current 4B integration validation

The [integration record](experiments/qwen4b-default-integration.json) covers the
current compact profile. One fixed-source smoke generated a fresh three-claim
answer and review in 37.24 s including runtime load and grammar work. The nested
native draft call took 15.27 s for 176 tokens and review 9.96 s for 100 tokens;
these call timings exclude grammar compilation and must not be added to the
inclusive wall time.

Two sequential live requests each fetched up to 100 current PubMed candidates:

| Metric | Cold passage cache | Reused passage cache |
| --- | ---: | ---: |
| Command wall time | 68.62 s | 76.66 s |
| Request time | 68.03 s | 75.82 s |
| Answer stage, including load and grammar work | 44.11 s | 62.34 s |
| PubMed download | 4.59 s | 7.88 s |
| Passage preparation | 15.65 s | 1.19 s |
| Unique passage inputs | 279 | 276 |
| Cache hits / newly encoded inputs | 0 / 279 | 274 / 2 |

Both requests completed with automatic `partial` status, three visible claims
and three checked exact-source citations. Each loaded one runtime, generated a
fresh draft and support review, and closed its worker. PubMed snapshots and
selected evidence changed. Cache reuse refers only to passage vectors; these
are execution and source-binding observations, not a paired model-speed
comparison. The reused-cache request was slower overall. No new human quality
assessment of these outputs was performed; automatic status and exact citations
do not establish claim support or answer completeness.

Validation discovered 377 main-environment unit tests, with seven optional
native-grammar tests skipped there. All 11 native structured-decoding CPU tests
passed in the isolated environment. Dependency checks passed in both environments.

## Historical 9B performance observations

The following observations predate the 4B default and compact contract. They
describe earlier 9B implementations, not the current default's integrated latency.
The [historical performance record](experiments/native-mlx.json) retains the
measurements and scopes.

On the Air M3 (24 GiB), a matched 2,107-token input and 64-token output cap took
32.46 s in Transformers/Metal versus 16.68 s in native 9B with linear coverage.
Startup wall time was 24.06 versus 5.35 s, including parent tokenizer/import work.
The native worker alone started in 2.31–3.46 s across six replays. Input-ID hashes
and EOS policy matched; the deliberate cap was a timing probe, not an answer.

| Development case | Historical Transformers pipeline | Historical native 9B linear pipeline | Automatic status |
| --- | ---: | ---: | --- |
| p03 / 30 candidates | 132.47 s | 61.80 s | partial |
| p06 / 100 | 189.86 s | 83.52 s | partial |
| p03 / 100 | 184.88 s | 68.76 s | validation_failed |
| p06 / 30 | 184.63 s | 88.56 s | validation_failed |
| p03 / 300 | 168.78 s | 95.14 s | partial |
| p06 / 300 | 208.77 s | 62.13 s | insufficient_evidence |

These fixed-source runs exclude PubMed refetch/passage preparation. They preserved
Dense ranking, sources and evidence payload, generated fresh answers/reviews and
reaped their workers. Output lengths and review prompts can differ. A quick
format failure is not a quicker valid answer, and automatic statuses are not
human correctness labels. Native 9B had two withheld format/enum failures versus
one historical Transformers failure.

The historical all-4-bit 9B artifact occupied 5.06 GB versus 7.98 GB for linear
coverage. Native allocator peaks were 6.58 versus 9.51 GB. Its pipeline observations
were 44.60–84.85 s, with three withheld format failures. These are decimal bytes;
overlapping memory counters must not be summed. Sequential observations do not
establish a latency distribution or isolate thermal/swap effects.

Two earlier fresh native **9B** live requests fetched 100 articles and produced
partial drafts. Command wall times were 65.37 s cold and 54.69 s with passage
cache reuse; PubMed was fetched again in both. Passage preparation fell from
11.46 to 0.05 s, while the 9B draft/review stage including startup took about
47–48 s. These are historical observations with changing live sources, not
current 4B guarantees.

The first unconstrained community 4B probes violated the three-claim contract and
were withheld before review. Their draft-only timings do not measure a completed
answer chain. Subsequent named-review research observed a 2.27x median paired
reviewer-call speed ratio on eleven identical input sets. That excludes startup,
grammar compilation, draft generation, PubMed and BGE; it is not an end-to-end
speedup or a quality-equivalence result.

## Reproduce fixed-source timing

```sh
.venv/bin/python scripts/benchmark_runtime.py \
  --source-run results/live/20261110/paired-live-v1/p03-30-cold/run.json \
  --answer-model 4b --engine mlx \
  --output results/performance/new-4b-replay \
  --summary results/performance/new-4b-summary.json
```

Only the explicitly supplied saved source runs are read. A new draft and its
newly constructed review execute through the public pipeline; this is not fresh
PubMed search or an answer-quality evaluation. Pass `--answer-model 9b` for the
9B comparison. Repeating `--source-run` repeats a case. Each replay creates a new
worker, but does not reset OS/model caches or thermal state. Failed validation
has a nonzero exit code and stays in the measurement inventory.

Native call seconds include synchronized generation and cache cleanup. TTFT is
the first generator yield, including MLX's one-token lookahead. Worker startup
contains import/load and must not be added to nested load measurements. Parent
RSS/MPS and child RSS/native peaks have different overlapping scopes; they are
not system-wide memory peaks.
