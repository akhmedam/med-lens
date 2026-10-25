# Local BGE embeddings

`BAAI/bge-base-en-v1.5` now encodes prepared windows on MPS or an explicitly
selected CPU. The checkpoint revision matches the tokenizer:
`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`. This stage creates vectors; exact
cosine retrieval, article-level aggregation, and ranking evaluation come next.

```sh
python -m pip install -e '.[embeddings]'
python scripts/embed_windows.py model --output .cache/models/bge-base-en-v1.5
python scripts/embed_windows.py build data/windows/pilot-v1-002/p01 \
  --model .cache/models/bge-base-en-v1.5 \
  --tokenizer .cache/tokenizers/bge-base-en-v1.5 \
  --output data/embeddings/pilot-v1/p01
python scripts/embed_windows.py inspect data/embeddings/pilot-v1/p01
```

The `model` command downloads only `config.json` and `model.safetensors` and
records their hashes. Download once, then build each query's windows offline.
Existing output directories are refused. `build` verifies assets, uses local
files only, and disallows remote model code. It defaults to MPS, batch size 16;
choose `--device cpu` or another `--batch-size` explicitly. MPS requires a working
backend and rejects `PYTORCH_ENABLE_MPS_FALLBACK=1`; no automatic CPU switch occurs.
`inspect` verifies artifacts without loading Torch, Transformers, or model weights.

## Encoding policy

Use float32, eager attention, evaluation mode, and inference mode. The passage
input IDs come directly from the [window artifact](embedding-windows.md): they
are not decoded or retokenized. Batches are right-padded with token ID 0 and use
an attention mask; token-type IDs are all zero. Every unpadded input must fit
512 tokens and start/end with BGE's CLS/SEP IDs.

The embedding is `last_hidden_state[:, 0, :]`, followed by L2 normalization.
Neither mean pooling nor the model's pooler output is used. Output vectors have
768 float32 components. Nonfinite and zero-norm outputs fail instead of entering
the artifact. Unit-norm vectors will support cosine similarity through a dot
product; no vector index or search service is introduced here.

Queries use the same encoder with this prefix, including the trailing space:

```text
"Represent this sentence for searching relevant passages: "
```

`BGEEncoder.encode_queries(texts)` preserves the original query after that prefix.
Documents receive no instruction. Queries exceeding 512 tokens including prefix
and special tokens fail without silent truncation. This is a recorded baseline
policy from the [pinned model card](https://huggingface.co/BAAI/bge-base-en-v1.5/blob/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/README.md),
not a prompt choice tuned on the pilot judgments.

## Saved artifacts

Each new output directory contains:

- `vectors.npy`: float32 matrix in source window order, loaded with `allow_pickle=False`.
- `rows.jsonl`: ordered window ID, PMID, and text-version ID for each vector.
- `windows-manifest.json`: exact copy of the validated source manifest.
- `manifest.json`: input/output hashes, encoder settings, code hash, artifact ID,
  package versions, and measured encoding time/allocations.

The encoder tokenizer must match the window tokenizer metadata exactly. Replay
checks artifact identity, hashes, row counts/IDs, source identity, shape, dtype,
finite values, and unit norms. Later retrieval must retain this mapping rather
than treating vector positions as article IDs. Empty input produces a valid
`(0, 768)` array. Source text, weights, and vectors remain ignored local files.

## M3 Pro check

The [recorded run](experiments/bge-embeddings.json) on 2026-10-25 encoded all
975 pilot windows on MPS in 21.61 seconds, plus 10 queries in 0.14 seconds.
The ten window matrices contain 2,995,200 bytes of float32 payload, about
2.86 MiB, excluding file headers and mapping metadata. Checkpoint weights are
437,955,512 bytes. Timings exclude download, artifact I/O, and asset verification;
the first window batch includes runtime warmup. Model materialization and transfer
after verification took 0.37 seconds in this cached run.

All vectors passed finite-value and unit-norm checks. Two probe inputs of 6 and
418 tokens were encoded separately, together with padding, repeatedly on MPS,
and on CPU. Maximum absolute differences were `1.75e-7` for batch/padding,
zero for the repeated MPS run, and `2.48e-7` between CPU and MPS. These numerical
checks cover two probes, not every sequence or future runtime version.

Process peak RSS was about 1.32 GiB, including model validation and CPU reference
checks. Sampled MPS driver allocations reached about 2.02 GiB across window/query
batches. These counters overlap and must not be added; sampled allocation is not
a system-wide RAM peak or a complete transient GPU peak.

All 90 offline project tests pass. Tensor tests verify CLS pooling, normalization,
padding masks, batch/order consistency, and input rejection. Artifact tests verify
source mapping, corruption checks, tokenizer mismatch, and failure without a
published output. Tests never download weights; the separate real-model run
checks actual BGE inference. Retrieval quality has not been measured for BGE yet.
