# MedCPT pair scorer

The first cross-encoder implementation uses
[`ncbi/MedCPT-Cross-Encoder`](https://huggingface.co/ncbi/MedCPT-Cross-Encoder)
at revision `71caf65d4927987813984f54c284405a13fcca49`. It has 109,483,009
parameters and runs in float32 on explicitly selected CPU or MPS. No Qwen model
is loaded. This stage verifies inference and input handling, not ranking quality.

## Input contract

`MedCPTScorer.score` takes ordered triples `(question, title, excerpt)`. The
second sequence is the unchanged title, a newline and the unchanged excerpt.
The tokenizer constructs `[CLS] question [SEP] document [SEP]` with attention
masks and segment IDs. Raw single-class logits are returned in the same order;
higher means more relevant, not a calibrated probability. No sigmoid or softmax
is applied. The [official model card](https://huggingface.co/ncbi/MedCPT-Cross-Encoder)
and [pinned configuration](https://huggingface.co/ncbi/MedCPT-Cross-Encoder/blob/71caf65d4927987813984f54c284405a13fcca49/config.json)
describe the output and 512-position architecture.

All tokens in the complete pair count toward the 512-token limit. The scorer
validates the entire request before the first forward pass and rejects an
oversized pair with its measured length. It never truncates, splits or silently
skips source text. Invalid logits, nonfinite scores and unexpected devices fail
explicitly. Title-only scoring is allowed for retrieval; it does not supply
abstract evidence for answer generation.

MedCPT's special IDs are CLS=2, SEP=3 and PAD=0, unlike BGE's CLS=101/SEP=102.
Its published tokenizer has an effectively unbounded `model_max_length` field;
the model configuration, not that field, establishes the enforced 512 limit.
The two tokenizers are independently pinned and checked.

Assets are downloaded only by an explicit download operation. The local manifest
pins all nine model/tokenizer/documentation files by SHA256; inference verifies
them and uses `local_files_only=True`, `trust_remote_code=False`. Upstream
publishes PyTorch `.bin` weights at this revision, so loading explicitly uses
`weights_only=True`. Runtime metadata records versions, parameter count, input
format, precision and device. The model's public-domain license is preserved
with the cached assets.

## Run

```sh
python -m pip install -e '.[reranking]'
python scripts/probe_medcpt.py --download \
  --output results/preflight/medcpt-example.json \
  --summary results/preflight/medcpt-example-summary.json
```

Use fresh report paths. The default asset directory is
`.cache/models/medcpt-cross-encoder`. A repeated `--download` verifies a complete
local manifest; an interrupted download without a manifest can resume. With
assets present, omit `--download` to keep inference local.

```python
from medlens.reranking.medcpt import MedCPTScorer

scorer = MedCPTScorer('.cache/models/medcpt-cross-encoder', device='mps', batch_size=4)
result = scorer.score([
    ('Does the intervention reduce symptoms in adults?',
     'Exact article title', 'Exact source excerpt'),
])
scores = result['scores']
```

MPS availability is checked; CPU fallback is not silently enabled. Select CPU
explicitly if needed. The same weights, float32 precision and eager attention
are used for the device comparison.

## Development probe

The runner validates all stored windows from the ten pilot datasets against the
source sections. It does not load qrels or the held-out questions. It audits the
full pair token lengths, then selects 32 fitting pairs by length quantiles,
ordered by length. For each device and batch size it warms that exact workload
once, then records three repetitions. This is a small, length-ordered inference
probe, not a full reranking request or representative service latency benchmark.

The [initial probe](experiments/medcpt-runtime-initial.json) guided the starting
batch size of four. The [confirmation probe](experiments/medcpt-runtime.json)
records the final scorer version. Batch sizes 1, 4, 8 and 16 are compared on both
CPU and MPS. A separate synthetic 512-token pair completes a forward pass; a
513-token pair is rejected. The raw local report keeps individual scores and
source pair mappings; public summaries omit article text.

On the M3 Pro, the confirmation run measured these median score times for the
same 32 pairs (three repetitions, one workload warm-up per configuration):

| Batch size | CPU | MPS |
| --- | ---: | ---: |
| 1 | 1.405 s | 0.483 s |
| 4 | 1.251 s | 0.404 s |
| 8 | 1.483 s | 0.467 s |
| 16 | 2.012 s | 0.670 s |

Batch four is the starting default, not an optimum established for every workload.
Maximum CPU/MPS score drift was 0.00002337; all batch comparisons passed the stated
tolerances. On five additional development pairs, the scorer matched a direct
Transformers tokenizer-padding/model call exactly on CPU and MPS; see the
[reference check](experiments/medcpt-reference-check.json). The process peak RSS
across both devices was 1.74 GiB. At batch four,
after-batch MPS samples were approximately 418 MiB of tensors and 1.02 GiB of
driver allocation. These are separate observations, not additive memory costs.

All 975 pilot question/window pairs fit without changing the BGE window
boundaries: complete inputs range up to 485 MedCPT tokens. This is an observation
on these questions, not a guarantee for future inputs. A longer question or new
text may need different windows. Any later window change must be versioned and
shared by both MedCPT modes and the matching Dense comparison.

Process peak RSS includes setup and both sequential device runs; it is not a
per-device peak. MPS tensor/driver values are samples after batches and can miss
transient allocations. They overlap unified-memory accounting and must not be
added to RSS. Load time includes file verification; score time excludes model
loading and warm-up. Numerical agreement is checked at absolute tolerance 1e-4
across batch sizes and 1e-3 between devices; it is not bitwise equality or a claim
that rankings cannot change for near-tied scores.

## Next experiment

Day 16 adds article aggregation and the Dense → MedCPT/direct MedCPT modes. The
current `ask.py --rerank` still calls Qwen; no MedCPT ranking score or quality
gain is reported yet. Day 17 evaluates relevance and expands incomplete pools;
Day 18 measures actual candidate-pool workloads. The runtime batch size can be
revisited there using development data.

The [held-out questions and procedure](../benchmarks/heldout-v1/README.md) are
frozen before those experiments. Their twelve questions have no retrieved
snapshots, rankings or judgments yet. The active pilot qrels v2 profile is
unchanged. Test-split collection and final confirmation remain future work.
