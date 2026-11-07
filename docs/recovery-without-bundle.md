# Recovery without the Day 19 bundle

The original archive parts and old Mac are unavailable. Public pinned models can
be downloaded exactly; live PubMed searches produce a **new corpus**, not a replay
of the lost pilot. Keep the frozen benchmark profiles and historical reports.
Never substitute rebuilt data into historical qrels or claim reproduced metrics.

## Environment and assets

On macOS arm64, from the checkout with Python 3.11:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-macos-arm64-day19.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python scripts/prepare_models.py --download \
  --output results/recovery/setup-assets.json
```

The asset manifest pins SHA-256 and sizes for BGE, its saved tokenizer, MedCPT,
Qwen and the original version-1 Metal kernel build. Conflicting existing payloads
fail before downloading; metadata is never overwritten. The script uses public
assets without a Hugging Face token and HTTP transport for model downloads.
Qwen needs about 19.3 GB of downloaded weight shards plus caches and working space.
The native kernel requires the recorded Torch 2.14/macOS arm64 environment.
Run the same command without `--download` for offline asset verification. Always
choose a fresh report path. `HF_HOME` for inference must point to this checkout's
`.cache/huggingface`; the scripts set it when it is absent.

The normal kernel resolver retains publisher/signature checks and can require
network access even with locally cached model weights. No CPU inference fallback,
allocator-limit override, model substitution or quantization change is required.

## Rebuild a separate corpus

Use a new version name for each fresh PubMed collection:

```sh
.venv/bin/python scripts/recover_corpus.py --version recovery-NEW --fetch \
  --output configs/collections/recovery-NEW.json
.venv/bin/python scripts/check_resume.py \
  --collections configs/collections/recovery-NEW.json \
  --output results/recovery/inputs-NEW.json
```

This follows the ten development questions and 30-candidate plan, stores raw
ESearch/EFetch snapshots, parses them, builds section-bounded BGE windows and
normalized 768-dimensional vectors on MPS. Existing complete artifacts are
verified and reused; without `--fetch`, missing snapshots fail instead of making
live requests. The frozen `pilot-v1` namespace and symlinked corpus paths are
rejected. Interrupted partial directories are preserved for inspection; they are
not silently replaced. Use a fresh version to recover from incomplete artifacts.

The portable collection profile binds question order/hash and each artifact ID.
`check_resume` without `--collections` still checks the original frozen corpus;
its failure after bundle loss is expected. `inputs_ready` verifies inputs and
dependencies, not inference or scientific answer quality.

## Use the recovered collection

The 2026-11-06 rebuilt collection already present on the recovery machine uses:

```sh
.venv/bin/python scripts/ask.py data/parsed/recovery-20261106/p06 \
  --windows data/windows/recovery-20261106/p06 \
  --embeddings data/embeddings/recovery-20261106/p06 \
  --query "Does cognitive behavioral therapy reduce anxiety symptoms in adults with generalized anxiety disorder?" \
  --generate --output-dir results/requests/recovered-gad-NEW
open results/requests/recovered-gad-NEW/report.html
```

A recorded full CLI p01 request returned a partial draft; p06 demonstrated
output-limit withholding. The generation pilot records both successful and failed
cases in [the Air recovery record](experiments/air-recovery.json).

Keep `--rerank` absent from the primary Dense → evidence → Qwen workflow. A
completed request may withhold a malformed answer (exit 2); inspect answer status
separately from pipeline status. Search/excerpts remain usable after model errors.
Only one Qwen process should run at a time on the 24 GiB Air.

For a new developmental generation baseline, first save search-only requests
against every collection in the profile:

```sh
.venv/bin/python - <<'PY'
import json, subprocess, sys
from pathlib import Path
profile = json.loads(Path('configs/collections/recovery-20261106.json').read_bytes())
questions = {q['query_id']: q['text'] for q in map(json.loads,
    Path(profile['queries']['path']).read_text().splitlines())}
for row in profile['collections']:
    subprocess.run([sys.executable, 'scripts/ask.py', row['dataset'],
        '--windows', row['windows'], '--embeddings', row['embeddings'],
        '--query', questions[row['query_id']], '--output-dir',
        'results/requests/rebuilt-pilot-NEW/' + row['query_id']], check=True)
PY
.venv/bin/python scripts/generate_answers.py results/requests/rebuilt-pilot-NEW \
  --collections configs/collections/recovery-20261106.json --include-controls \
  --output results/generation/rebuilt-NEW.json \
  --summary results/generation/rebuilt-NEW-summary.json
```

The loader verifies actual datasets/windows/vectors, saved question/ranking and
exact source spans/metadata. Only completed Dense search-only requests qualify.
The pinned adolescent control must be present in selected evidence; otherwise
`--include-controls` fails explicitly. New live searches need not select it.
Reports declare `resource_scope=rebuilt-pilot` and no bound relevance judgments.
Original-context and rebuilt-corpus summaries are descriptive comparisons only;
no paired improvement delta is justified. Held-out questions remain unused.

## What remains lost

The seven recovered Day 19 sidecars preserve manifest/validation evidence, but
none of the ten archive parts survived. Historical raw source snapshots, vectors,
rankings, full prompts and Qwen outputs cannot be reconstructed byte-for-byte
from aggregate reports. Fresh downloads and inference establish new identities.
Keep a separate backup of the ignored `data/`, `.cache/` and raw `results/` trees,
together with the exact checkout/manifest; Git alone does not back up these assets.


## Save a new transfer bundle

After all runs finish and the checkout is committed, create and verify a new
complete bundle with the existing transfer tool:

```sh
.venv/bin/python scripts/transfer_project.py pack transfer/recovery-NEW
.venv/bin/python scripts/transfer_project.py verify transfer/recovery-NEW
```

The new manifest binds tracked recovery configs as well as source/benchmark
files. Copy **the whole directory, including every archive part**, to a separate
backup location. A manifest or validation sidecar alone cannot restore payloads.
A bundle left on the same laptop is only a local transfer copy.
