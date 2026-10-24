# BGE passage windows

This stage prepares model inputs for `BAAI/bge-base-en-v1.5`. It does not compute
embeddings or change retrieval rankings. Only tokenizer assets are downloaded;
model weights and embedding inference belong to the next stage.

```sh
python -m pip install -e '.[tokenization]'
python scripts/prepare_windows.py tokenizer \
  --output .cache/tokenizers/bge-base-en-v1.5
python scripts/prepare_windows.py build data/parsed/pilot-v1/p01 \
  --tokenizer .cache/tokenizers/bge-base-en-v1.5 \
  --output data/windows/pilot-v1-002/p01
python scripts/prepare_windows.py inspect data/windows/pilot-v1-002/p01
```

Use a new output directory for each build. The `tokenizer` command can access
Hugging Face; `build` loads local assets only and `inspect` needs neither the
network nor Transformers. Downloading once is sufficient for all pilot queries.
The fixed model revision is `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`.
Saved tokenizer files are hashed and verified before loading. The manifest also
records the serialized fast-tokenizer backend hash and library versions.

## Window policy

The input is a single BERT sequence: `[CLS]`, title token IDs, a slice of primary
abstract section token IDs, then `[SEP]`. There is no passage instruction and no
extra separator between title and section. Case handling and WordPiece splitting
come from BGE's tokenizer, not BM25 preprocessing. Section headings are metadata,
not additional model tokens.

The defaults are 384 **body** tokens and 64 tokens of overlap. Windows never cross
section boundaries. Short sections remain short; they are not combined with
neighbors. The complete input, including the title and two special tokens, must
fit 512 tokens. A long title reduces body capacity; subsequent windows keep the
configured overlap. If even the full title or required overlap cannot fit, the
build fails explicitly instead of truncating source text. Override body size and
overlap with `--body-tokens` and `--overlap`.

Every source section token is covered. Final windows stop at the last source
token, with no redundant extra window when the previous one already reaches it.
An absent or empty primary abstract produces a title-only input if a title is
available. Other-language abstracts and book titles are not silent fallbacks.
Records without any title or primary-abstract tokens produce no input and are
listed in `empty_pmids`.

## Source spans and downstream use

Each JSONL row retains PMID, text-version ID, source provenance, section index,
label and XML path, half-open token and character spans, the exact source excerpt,
the title, input IDs, and token counts. Character offsets refer to the parsed
section's normalized `text`, not raw XML byte offsets. The original snapshot is
still the authoritative raw source.

**Use the stored `input_ids` for embedding inference.** A token window may start
inside a WordPiece-split word. Retokenizing or decoding its displayed excerpt
can change token IDs and invalidate the window budget/overlap. The excerpt is
for display and source grounding; it is sliced directly from source text,
without decoding tokens or rewriting the article. An encoder will add batch
padding and an attention mask; no padding is stored in this artifact.

Window IDs include source content/provenance, spans, token IDs, and configuration.
The artifact ID additionally binds dataset and article-file hashes, tokenizer
identity, and builder code. JSONL and manifest are created in a new directory;
replay checks hashes, artifact/window IDs, counts, and model input lengths.
Build validates the parsed dataset first. Text and token artifacts stay in
ignored `data/`; tokenizer assets stay in `.cache/`.

## Pilot check

On 2026-10-24, all 300 candidate memberships from the ten-query pilot produced
975 windows. Seven were title-only; no article membership lacked an input.
Twenty-four sections needed multiple windows. The largest full input was
418 tokens, below the 512-token limit. These counts include articles appearing
under multiple queries and are not counts of unique PubMed papers.

An offline audit with the real BGE tokenizer checked every section's token
coverage, exact excerpt offsets, each stored input sequence, 64-token overlap
where consecutive windows exist, and the input limit. The
[aggregate report](experiments/embedding-windows.json) contains versions, hashes,
and per-query counts without source article text.

All 82 offline tests pass. New tests cover boundary lengths, overlap, section
isolation, Unicode source spans, title budgets, absent text, deterministic
artifacts, corruption, and CLI replay. Unit tests use a small tokenizer double
and never download assets; the saved-pilot audit separately exercises BGE's
actual WordPiece tokenizer.

The [pinned BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5/blob/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/README.md)
documents the 512-token context and passage/query instruction distinction.
