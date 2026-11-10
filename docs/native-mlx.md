# Native MLX inference on Apple Silicon

The live and local request CLIs select an explicit Qwen backend. Native MLX runs
in a separate Python process/environment; BGE and the original Hugging Face
answer tokenizer remain in the existing environment. The model remains the
pinned Qwen3.5-9B revision. Runtime and quantization arithmetic are different;
matching model names, prompts and citations do not establish answer equivalence.

## Prepare the isolated environment

From the checkout, after restoring the pinned original assets with
`scripts/prepare_models.py`:

```sh
.venv/bin/python -m venv .cache/venvs/mlx
.cache/venvs/mlx/bin/python -m pip install -r requirements-mlx-macos-arm64.lock
.venv/bin/python scripts/prepare_mlx.py
```

The existing main environment and its lock remain unchanged. MLX requires an
Apple GPU; a CPU-only or incompatible setup fails explicitly. No dependency,
backend, model or stale PubMed result is silently substituted.

Preparation hashes every pinned source file, converts offline with MLX-LM 0.32.2,
checks the saved tensor layout/dtypes/config, hashes all produced files and
publishes `medlens-source.json` only for a completed artifact. Output must be
fresh. Failed preparation leaves a staging directory/status for inspection;
rerun after fixing the cause, without reusing or overwriting a partial output.
There is no extra model download when the original Qwen weights are present.

The default artifact is `.cache/models/qwen3.5-9b-mlx-linear-4bit-g64`.
Both available policies use affine 4-bit/group-64 quantization. The policy is
recorded in the manifest. `linear-only-except-lm-head` retains FP16 embeddings
and output head, matching the original Metal coverage (the default); `all-linear-and-embedding`
quantizes them too and is an explicitly different experiment. Nonquantized
floating parameters use FP16, except native Qwen `A_log`, which stays FP32.
The original Transformers path loaded FP16, so even the closer coverage is not
a bitwise-equivalent conversion. Greedy output sequences can differ.

Use `--coverage` and `--output` to prepare a second policy in a separate directory.
Do not edit a manifest to relabel existing weights.

## Run and inspect

```sh
.venv/bin/python scripts/ask_live.py \
  --query "Does exercise reduce depressive symptoms in adults with depression?" \
  --limit 100 --output-dir results/requests/native-example
```

`--engine transformers` explicitly selects the former MPS/Metal implementation.
`--mlx-model-dir` chooses another prepared native artifact, and `--mlx-python`
chooses its isolated interpreter. Retrieval-only/empty-source requests do not
require Qwen assets. Startup/inference failures retain completed source evidence.
The saved request records engine policy, manifest hash, artifact ID, packages,
precision, stop policy and runtime source hashes. Actual token IDs are hashed
separately from source messages. Model outputs, sources and weights remain local.

One worker loads weights once for draft plus support review. Every call gets a
fresh unquantized KV/recurrent cache; weights persist, state from a different
prompt does not. The original pinned HF tokenizer constructs input IDs and
decodes output IDs, avoiding tokenizer drift between environments. Input and
output IDs, budgets, session nonce and response identity are checked on both
sides of a bounded, deadline-controlled JSON-lines protocol. Stderr is separate
from the protocol. Success, errors, timeout and interruption all close/reap the
worker; termination escalates to process-group kill when necessary.

Both backends now explicitly stop on either source-config end-of-text (248044)
or chat end (248046). Former Transformers calls inherited only source-config
EOS. This is a named stopping-policy change, not a hidden runtime equivalence
claim. A terminal EOS counts toward the output count; reaching the cap remains
a validation failure even if the last token is EOS. Answer budgets remain
6,500 input / 768 draft / 512 review tokens; optional listwise retains its
12,000 input / 128 output budget. There is no context shortening, answer replay,
score repair, JSON repair or removal of the support review.

Raw NaN, positive infinity and fully masked score rows fail before sampling.
Allowed negative-infinity masks remain allowed. Startup validates the canonical
manifest identity, every file size and config SHA256. It does not hash gigabytes
on every request; `MlxRuntime(verify_weights=True)` performs a full file-hash
verification when required. Preparation always hashes complete files.

## Observed performance

On the Air M3 (24 GiB), a matched 2,107-token input and 64-token output cap took
32.46 s in current Transformers/Metal versus 16.68 s in native MLX with the
selected linear coverage. Startup wall time was 24.06 versus 5.35 s, including
the parent tokenizer/import costs; native worker startup alone in the six
replays was 2.31–3.46 s. Both calls produced 64 tokens with identical input-ID
hashes and EOS policy. The cap was deliberate for timing, not an accepted answer.

Fixed-source full-pipeline observations:

| Development case | Historical Transformers pipeline | Native linear pipeline | Native automatic answer status |
| --- | ---: | ---: | --- |
| p03 / 30 candidates | 132.47 s | 61.80 s | partial |
| p06 / 100 | 189.86 s | 83.52 s | partial |
| p03 / 100 | 184.88 s | 68.76 s | validation_failed |
| p06 / 30 | 184.63 s | 88.56 s | validation_failed |
| p03 / 300 | 168.78 s | 95.14 s | partial |
| p06 / 300 | 208.77 s | 62.13 s | insufficient_evidence |

These exclude PubMed refetch and passage preparation. Every run preserved the
same Dense ranking, selected sources and evidence payload, used fresh model
calls, and reaped its worker. Generated sequences/review inputs can differ;
this is not equal-quality latency or a speed distribution. In particular, an
early format refusal is not a faster valid answer. The six native cases had
two withheld format/enum failures versus one in the six historical cases. No
correctness labels or human relevance metrics were computed.

The all-4-bit coverage occupied 5.06 GB on disk versus 7.98 GB for selected linear;
observed native allocator peaks were 6.58 versus 9.51 GB. These are decimal bytes
and overlapping allocator/process counters are not summed. All-4-bit pipeline
observations were 44.60–84.85 s, with three withheld format failures. The closer
linear coverage is selected; neither count estimates a population failure rate.
Latency varied across sequential runs. A separate spot check observed 347.88 MiB
of existing system swap and no recorded macOS thermal/performance warning;
this does not establish whether throttling or swapping affected a given call.

An exploratory pinned community Qwen3.5-4B affine-4-bit artifact completed the
same 2,107-token/64-output-token case in 8.38 s. Four fixed-prompt observations
were 6.55–12.23 s. Its weights preserve BF16/FP32 and are not the locally converted
FP16 policy. The community SHA and weight SHA256 are recorded; the converter card
does not disclose an upstream official source revision. Common token mappings
match; seven additional audio tokens were not used and were rejected if generated.

Both fresh 4B drafts violated the existing maximum-three-claims contract (541/566
output tokens), so both were withheld before review. Their 26.75–29.89 s draft-only
latencies are not completed answer-chain timings. The smaller model is unselected
and is not exposed as a production backend. Raw exploratory script/outputs remain
local with hashes in the record. No model-generated relevance or correctness
labels were created.

Two fresh live requests through the new default CLI fetched 100 current articles
and completed with partial drafts and two verified visible citations each. Command
wall time was 65.37 s with a new passage cache and 54.69 s on reuse. All 272 passage
inputs were encoded cold; all 272 were cache hits warm. PubMed was fetched anew
in both runs, with distinct snapshots. Passage preparation fell from 11.46 s to
0.05 s; Qwen draft/review still took about 47–48 s including startup. Both runtimes
closed. Historical live command observations for the same development question
and candidate limit were 214.84/202.84 s; fresh result/provenance and generated
outputs may differ, so this is an observed end-to-end comparison rather than an
equal-quality causal estimate.

## Measure without judging answer quality

```sh
.venv/bin/python scripts/benchmark_runtime.py \
  --source-run results/live/20261110/paired-live-v1/p03-30-cold/run.json \
  --engine mlx --output results/performance/new-native-replay \
  --summary docs/experiments/new-native-replay.json
```

Only explicitly provided saved runs are read. This is fixed-source pipeline
replay, not fresh PubMed search. It includes query encoding, Dense/evidence,
model startup, a newly generated draft and its newly constructed review.
Raw reports stay in the output directory; the compact summary has identities,
counts, timings and integrity checks without clinical/source text. Repeating
`--source-run` explicitly repeats a case. Each replay creates a fresh worker;
model/OS caches and thermal state are not reset. A validation failure is reported
and has its own nonzero CLI exit code. It is not repaired or counted as a valid
answer.

See the [performance record](experiments/native-mlx.json) for measurements and
limits. Native call seconds include synchronized generation/cache cleanup.
TTFT is the first native generator yield and includes MLX's one-step lookahead.
Worker startup includes import/load and must not be added to its nested load
measurement. Parent RSS/MPS and child RSS/native allocator peaks have different,
overlapping scopes; they are not system-wide peaks and must not be summed.

Format/citation acceptance and same-model support review are not human quality
labels. New manual annotations and answer assessment remain independent work;
no correctness/relevance equivalence or improvement is claimed here.
