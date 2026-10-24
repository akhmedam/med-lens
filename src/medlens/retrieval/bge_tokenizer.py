"""Pinned BGE tokenizer assets; model weights are never requested here."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

MODEL_ID = "BAAI/bge-base-en-v1.5"
REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


def download_tokenizer(output: Path) -> dict:
    from transformers import AutoTokenizer

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
        use_fast=True, trust_remote_code=False, cache_dir=".cache/huggingface")
    output.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(output)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.iterdir()) if p.is_file()}
    source = {"model_id": MODEL_ID, "revision": REVISION, "files": files}
    (output / "tokenizer-source.json").write_text(json.dumps(source, indent=2) + "\n")
    return source


def load_tokenizer(directory: Path):
    from transformers import AutoTokenizer

    directory = Path(directory)
    source = json.loads((directory / "tokenizer-source.json").read_bytes())
    if source["model_id"] != MODEL_ID or source["revision"] != REVISION:
        raise ValueError("Unsupported tokenizer source")
    for name, expected in source["files"].items():
        if Path(name).name != name or (directory / name).is_symlink():
            raise ValueError("Invalid tokenizer asset path")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Tokenizer asset checksum mismatch")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True,
        use_fast=True, trust_remote_code=False)
    if (not tokenizer.is_fast or tokenizer.model_max_length != 512
        or tokenizer.num_special_tokens_to_add(pair=False) != 2
        or tokenizer.cls_token_id != 101 or tokenizer.sep_token_id != 102):
        raise ValueError("Expected BGE's fast BERT tokenizer with a 512-token limit")
    info = {**source, "backend_sha256": hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "transformers_version": version("transformers"), "tokenizers_version": version("tokenizers")}
    return tokenizer, info
