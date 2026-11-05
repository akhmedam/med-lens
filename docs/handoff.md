# Continue on the MacBook Air M3

Target: macOS arm64, M3, 24 GiB unified memory. Python 3.11. The source machine is
an M3 Pro with 36 GiB; Qwen inference on the Air has not been validated.

## Before leaving this machine

Merge the Day 19 PR, then clone `akhmedam/med-lens` on the Air. The PR contains
code, current dependency pins, both canonical pilot judgment versions, original
human rounds, assessment provenance/rationales, benchmark definitions, results
summaries and the pending [pipeline decision](pipeline-options.md).

Copy the **entire** `transfer/medlens-day19/` directory to the flash drive,
including `manifest.json` and every `part-*.tarpart`. A verified instance is
prepared at the end of Day 19. Parts are at most 2 GiB, so the flash drive does
not need to support files larger than 4 GiB or symlinks. A 32 GB drive is large
enough for this snapshot; use the manifest's exact `archive_bytes` as the check.
Do not concatenate parts or copy individual Hugging Face snapshot directories.

The bundle includes:

- all local `data/`: raw PubMed snapshots, parsed records, windows, vectors,
  historical/user annotations and annotation evidence/pages;
- all local `results/`: raw rankings, model outputs, contexts, generation caches,
  request reports and measured runs;
- pinned BGE/MedCPT assets and tokenizer files;
- the project-local Hugging Face hub cache, including Qwen shards, shared blobs,
  tokenizer, snapshot metadata and cached Metal kernel build; relative symlinks
  are reconstructed after restore.

It excludes `.venv`, `.git`, credentials, environment files, Hugging Face token
files, download locks, incomplete downloads, Xet logs and Python bytecode. GitHub
login and any external API credentials must be configured separately if needed.
Public PubMed/model access does not require copying a token.

## Verified source bundle

The prepared bundle is 20,281,989,120 bytes (18.89 GiB): ten parts, 341 regular
files and 26 internal symlinks. Its manifest SHA-256 is
`a449b51fc711c4bfc51c9842e83e32d5cf94c6e2c2512aa00393fbdf96cf2c90`.
The [verification record](experiments/handoff-day19.json) pins every part and
records a clean-clone install, 200 passing tests, exact frozen-evaluation replay
and an offline search returning ten articles/six excerpts without Qwen weights.
These checks ran on the source M3 Pro, not the target Air. The `validation/`
sidecar carries the raw checks/reports; restore does not need to install it.

## Restore on the Air

Run these commands from the new clone. Substitute the flash drive's actual path
for `/Volumes/FLASH/medlens-day19`; the bundle may also be copied to local disk.
Python's standard library handles restore before dependencies are installed.
Internet is needed for Python/packages and may be needed for Metal publisher
checks even with the model weights restored.

```sh
git clone https://github.com/akhmedam/med-lens.git
cd med-lens
python3.11 scripts/transfer_project.py verify /Volumes/FLASH/medlens-day19
python3.11 scripts/transfer_project.py restore /Volumes/FLASH/medlens-day19
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-arm64-day19.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python scripts/check_resume.py --output results/handoff/air-input-check.json
.venv/bin/python -m unittest discover -s tests
```

Use Python 3.11 and the Day 19 lock; the older Day 1 lock lacks `bm25s`. The
virtual environment must be newly created on this machine. The source Python was
3.11.15 on macOS 26.7.1. A different macOS/Python patch version is not automatically
an exact binary reproduction; record it in the preflight report.

Restore verifies every part first, rejects path escapes and symlinks outside the
bundle, and checks code/config hashes against the clone. Existing artifact files
are accepted only if identical; conflicts are refused rather than overwritten.
Code/config hashes permit later documentation-only commits. If a future code
change causes a mismatch, check out the bundle's `source_commit` in a suitable
checkout and restore there before upgrading. An interrupted restore can leave
complete files; repeat restore validates them. A corrupted regular file being
written is removed on failure. Keep the source/bundle until the new-machine checks
pass; the original machine is not modified by restore.

Allow roughly 20 GB for the restored assets plus the virtual environment. If
copying the bundle to the Air's SSD before restore, temporarily allow roughly
twice the artifact size. Keep the flash drive intact until restoration is checked.

## Staged verification

`check_resume.py` verifies package pins, label identities, BGE/MedCPT checksums,
all ten dataset/window/vector collections, the cached Qwen tokenizer and resolving
weight shards. It does not load Qwen weights or prove inference fits in 24 GiB.
Successful output is `inputs_ready`.

Replay the original comparison without network/model inference, choosing fresh
output paths:

```sh
.venv/bin/python scripts/evaluate_medcpt.py \
  --output results/handoff/air-medcpt-replay.json \
  --summary results/handoff/air-medcpt-replay-summary.json
```

Check MPS/package compatibility without loading Qwen weights:

```sh
.venv/bin/python scripts/smoke_inference.py \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --output results/handoff/air-mps-preflight.json
```

Then test the existing search/evidence path. This loads BGE and the Qwen tokenizer,
without Qwen weights:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/ask.py data/parsed/pilot-v1/p06 \
  --windows data/windows/pilot-v1-002/p06 \
  --embeddings data/embeddings/pilot-v1/p06 \
  --query "Does cognitive behavioral therapy reduce anxiety symptoms in adults with generalized anxiety disorder?" \
  --output-dir results/requests/air-search-001
```

If the user chooses to pursue Qwen drafts, first measure loading/short inference
on the Air, then a real bounded request. Close unrelated heavy applications and
record memory pressure/swap and elapsed time. Do not force `HF_HUB_OFFLINE=1` for
these Qwen runs; cached kernels can still need a publisher/version lookup.

```sh
.venv/bin/python scripts/smoke_inference.py --run \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --output results/handoff/air-qwen-short.json

.venv/bin/python scripts/ask.py data/parsed/pilot-v1/p06 \
  --windows data/windows/pilot-v1-002/p06 \
  --embeddings data/embeddings/pilot-v1/p06 \
  --query "Does cognitive behavioral therapy reduce anxiety symptoms in adults with generalized anxiety disorder?" \
  --generate --output-dir results/requests/air-draft-001
```

A short smoke success does not prove the full 6500-input-token / 4000-evidence-token
budget fits. A `validation_failed` draft is withheld; exit 2 is an expected strict
output failure to investigate, not a reason to loosen gates. Retrieval can remain
usable if Qwen fails. Do not enable CPU fallback or raise allocator limits to mask
an unsupported configuration.

## Work to resume

1. Read `docs/pipeline-options.md` and ask the user for retrieval and answer-mode
   choices. Both remain pending; Dense is a recommendation, not a new accepted
   selection. Current `ask.py --rerank` still means historical Qwen reranking.
2. Integrate the selected path. MedCPT requires explicit raw-logit context support
   and model lifecycle integration before it can feed `ask.py` generation. Do not
   clamp logits to the existing cosine range.
3. Continue Days 20–21: format reliability, age/population/comparator/outcome
   fidelity, contradictions and semantic answer checks. Four of ten Day 13
   questions had visible drafts; same-model review missed an age qualifier later.
4. Follow Days 22–28 in `docs/development-plan.md`. The terminal is the primary UI.
   Keep alternative code until comparison/final confirmation; remove losing paths
   for release after selection.
5. Keep `benchmarks/heldout-v1` frozen for Day 26. No held-out snapshots, judgments
   or model scores have been collected. Development labels are AI-assisted, with
   truthful provenance and original human rounds preserved.

Use feature branches without a `codex/` prefix, push a PR, and let the user merge
in GitHub. The actual source checkout was `~/Projects/MedLens`; no absolute source
path is required by the new-clone restore commands. Model weights, primary texts,
vectors and raw experiment outputs stay out of Git.
