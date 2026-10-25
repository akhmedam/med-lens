"""Local BGE CLS embeddings and checksummed vector artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from .bge_tokenizer import MODEL_ID, REVISION, load_tokenizer
from .windows import load_windows

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DIMENSION = 768


class EmbeddingError(ValueError):
    """Model inputs, vectors, or artifact identities are invalid."""


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def download_model(output: Path) -> dict:
    from huggingface_hub import snapshot_download

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    names = ("config.json", "model.safetensors")
    snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=list(names), local_dir=output)
    source = {"model_id": MODEL_ID, "revision": REVISION,
        "files": {name: _sha((output / name).read_bytes()) for name in names}}
    (output / "model-source.json").write_bytes(_json(source) + b"\n")
    return source


def query_input(tokenizer, text: str) -> list[int]:
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingError("Query must be nonempty text")
    ids = tokenizer(QUERY_INSTRUCTION + text, add_special_tokens=True,
        truncation=False, verbose=False)["input_ids"]
    if len(ids) > 512:
        raise EmbeddingError("Query plus instruction exceeds 512 tokens; no truncation applied")
    return ids


def normalized_cls(hidden):
    import torch

    vectors = hidden[:, 0, :].float()
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    if not torch.isfinite(vectors).all().item() or not torch.isfinite(norms).all().item() or (norms <= 0).any().item():
        raise EmbeddingError("CLS output contains nonfinite or zero-length vectors")
    return vectors / norms


def encode_sequences(sequences, model, *, device: str, batch_size: int = 16) -> tuple[np.ndarray, dict]:
    import torch

    if type(batch_size) is not int or batch_size <= 0:
        raise EmbeddingError("batch_size must be a positive integer")
    if device not in {"cpu", "mps"}:
        raise EmbeddingError("Choose cpu or mps explicitly")
    for ids in sequences:
        if (not 2 <= len(ids) <= 512 or ids[0] != 101 or ids[-1] != 102
            or any(type(token) is not int or not 0 <= token < 30522 for token in ids)):
            raise EmbeddingError("Expected unpadded BGE token IDs within 512 tokens")
    if device == "mps":
        torch.mps.synchronize()
    started = perf_counter()
    batches, max_tensor, max_driver = [], 0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            length = max(map(len, batch))
            ids = torch.zeros((len(batch), length), dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
                mask[i, :len(row)] = 1
            hidden = model(input_ids=ids, attention_mask=mask,
                token_type_ids=torch.zeros_like(ids)).last_hidden_state
            vectors = normalized_cls(hidden)
            if vectors.shape != (len(batch), DIMENSION):
                raise EmbeddingError("Expected a 768-dimensional BGE CLS vector")
            batches.append(vectors.cpu().numpy())
            if device == "mps":
                max_tensor = max(max_tensor, torch.mps.current_allocated_memory())
                max_driver = max(max_driver, torch.mps.driver_allocated_memory())
    if device == "mps":
        torch.mps.synchronize()
    result = np.concatenate(batches) if batches else np.empty((0, DIMENSION), dtype=np.float32)
    return result, {"encode_seconds": perf_counter() - started,
        "sampled_max_mps_tensor_bytes": max_tensor if device == "mps" else None,
        "sampled_max_mps_driver_bytes": max_driver if device == "mps" else None}


class BGEEncoder:
    def __init__(self, model_dir: Path, tokenizer_dir: Path, *, device: str = "mps", batch_size: int = 16):
        import torch
        from transformers import AutoModel

        if device not in {"cpu", "mps"} or type(batch_size) is not int or batch_size <= 0:
            raise EmbeddingError("Invalid device or batch size")
        if device == "mps" and (not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"):
            raise EmbeddingError("MPS must be available with CPU fallback disabled")
        model_dir = Path(model_dir)
        source = json.loads((model_dir / "model-source.json").read_bytes())
        if source["model_id"] != MODEL_ID or source["revision"] != REVISION or set(source["files"]) != {"config.json", "model.safetensors"}:
            raise EmbeddingError("Unsupported model source")
        for name, checksum in source["files"].items():
            if _sha((model_dir / name).read_bytes()) != checksum:
                raise EmbeddingError("Model checksum mismatch")
        self.tokenizer, self.tokenizer_info = load_tokenizer(tokenizer_dir)
        started = perf_counter()
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.float32,
            attn_implementation="eager")
        cfg = self.model.config
        if (cfg.model_type != "bert" or cfg.hidden_size != DIMENSION
            or cfg.max_position_embeddings != 512 or cfg.vocab_size != 30522):
            raise EmbeddingError("Unexpected BGE model architecture")
        self.model.to(device).eval()
        if device == "mps":
            torch.mps.synchronize()
        self.load_seconds = perf_counter() - started
        self.device, self.batch_size = device, batch_size
        self.info = {"model": source, "device": device, "dtype": "float32",
            "pooling": "last_hidden_state[:,0,:]", "normalization": "L2", "dimension": DIMENSION,
            "attention": "eager", "batch_size": batch_size, "pad_token_id": 0, "padding_side": "right",
            "query_instruction": QUERY_INSTRUCTION, "passage_instruction": None,
            "torch_version": version("torch"), "transformers_version": version("transformers"),
            "numpy_version": version("numpy")}

    def encode(self, sequences):
        return encode_sequences(sequences, self.model, device=self.device, batch_size=self.batch_size)

    def encode_queries(self, texts):
        return self.encode([query_input(self.tokenizer, text) for text in texts])


def build_embeddings(windows_dir: Path, encoder, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source, rows = load_windows(windows_dir)
    if source["tokenizer"] != encoder.tokenizer_info:
        raise EmbeddingError("Window and encoder tokenizers differ; rebuild explicitly")
    vectors, measurements = encoder.encode([row["input_ids"] for row in rows])
    validate_vectors(vectors, len(rows))
    buffer = io.BytesIO()
    np.save(buffer, vectors, allow_pickle=False)
    body = buffer.getvalue()
    mapping = b"".join(_json({key: row[key] for key in ("window_id", "pmid", "text_version_id")}) + b"\n" for row in rows)
    source_bytes = (Path(windows_dir) / "manifest.json").read_bytes()
    if json.loads(source_bytes) != source:
        raise EmbeddingError("Window manifest changed during encoding")
    identity = {"schema_version": 1, "source_window_artifact_id": source["artifact_id"],
        "source_manifest_sha256": _sha(source_bytes), "vectors_sha256": _sha(body),
        "rows_sha256": _sha(mapping), "encoder": encoder.info,
        "implementation_sha256": _sha(Path(__file__).read_bytes())}
    manifest = {**identity, "artifact_id": _sha(_json(identity)), "status": "complete",
        "row_count": len(rows), "dimension": DIMENSION, "measurements": measurements}
    output.mkdir(parents=True, exist_ok=False)
    (output / "vectors.npy").write_bytes(body)
    (output / "rows.jsonl").write_bytes(mapping)
    (output / "windows-manifest.json").write_bytes(source_bytes)
    (output / "manifest.json").write_bytes(_json(manifest) + b"\n")
    return manifest


def validate_vectors(vectors, count):
    if vectors.shape != (count, DIMENSION) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise EmbeddingError("Invalid vector shape, dtype, or finite values")
    if count and not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5, rtol=0):
        raise EmbeddingError("Vectors must have unit L2 norm")


def load_embeddings(directory: Path):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_bytes())
        if manifest["schema_version"] != 1 or manifest["status"] != "complete":
            raise EmbeddingError("Incomplete embedding artifact")
        for name, key in (("vectors.npy", "vectors_sha256"), ("rows.jsonl", "rows_sha256"),
                          ("windows-manifest.json", "source_manifest_sha256")):
            if (directory / name).is_symlink() or _sha((directory / name).read_bytes()) != manifest[key]:
                raise EmbeddingError("Embedding artifact checksum mismatch")
        identity = {key: manifest[key] for key in ("schema_version", "source_window_artifact_id",
            "source_manifest_sha256", "vectors_sha256", "rows_sha256", "encoder", "implementation_sha256")}
        if _sha(_json(identity)) != manifest["artifact_id"]:
            raise EmbeddingError("Embedding artifact identity mismatch")
        source = json.loads((directory / "windows-manifest.json").read_bytes())
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        rows = tuple(json.loads(line) for line in (directory / "rows.jsonl").read_bytes().splitlines())
        if (source["artifact_id"] != manifest["source_window_artifact_id"]
            or source["window_count"] != len(rows) or len(rows) != manifest["row_count"]
            or len({row["window_id"] for row in rows}) != len(rows) or manifest["dimension"] != DIMENSION):
            raise EmbeddingError("Vector-to-window mapping mismatch")
        validate_vectors(vectors, len(rows))
        return manifest, rows, vectors
    except EmbeddingError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EmbeddingError("Invalid embedding artifact") from exc
