"""Section-bounded token windows with exact source spans and title budgeting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from medlens.parsing.dataset import load_dataset


class WindowError(ValueError):
    """A record cannot be windowed without violating the configured limits."""


@dataclass(frozen=True)
class WindowConfig:
    body_tokens: int = 384
    overlap: int = 64
    max_input_tokens: int = 512

    def __post_init__(self):
        if any(type(x) is not int for x in (self.body_tokens, self.overlap, self.max_input_tokens)):
            raise WindowError("Token budgets must be integers")
        if not 0 <= self.overlap < self.body_tokens or not 2 < self.max_input_tokens <= 512:
            raise WindowError("Invalid window, overlap, or model input limit")


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def article_windows(article: dict, tokenizer, config: WindowConfig | None = None) -> list[dict]:
    config = config or WindowConfig()
    title = article.get("title") or ""
    title_ids = tokenizer(title, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    overhead = tokenizer.num_special_tokens_to_add(pair=False)
    if overhead != 2 or tokenizer.cls_token_id != 101 or tokenizer.sep_token_id != 102:
        raise WindowError("Expected BGE single-sequence CLS/SEP special tokens")
    if len(title_ids) + overhead > config.max_input_tokens:
        raise WindowError(f"PMID {article['pmid']}: title alone exceeds input budget")
    capacity = min(config.body_tokens, config.max_input_tokens - overhead - len(title_ids))
    result = []

    def append(section, ids, token_start=None, token_end=None, char_start=None, char_end=None):
        input_ids = [tokenizer.cls_token_id, *title_ids, *ids, tokenizer.sep_token_id]
        if len(input_ids) > config.max_input_tokens:
            raise WindowError("Internal error: window exceeds model input budget")
        row = {"pmid": article["pmid"], "text_version_id": article["text_version"]["text_version_id"],
            "kind": "abstract" if section is not None else "title",
            "section_index": section["index"] if section is not None else None,
            "section_label": section.get("label") if section is not None else None,
            "source_path": section["source_path"] if section is not None else None,
            "token_start": token_start, "token_end": token_end,
            "char_start": char_start, "char_end": char_end,
            "title": title, "text": section["text"][char_start:char_end] if section is not None else "",
            "title_token_count": len(title_ids), "body_token_count": len(ids),
            "input_ids": input_ids, "input_token_count": len(input_ids),
            "provenance": article["provenance"]}
        row["window_id"] = _sha(_json({"row": row, "config": asdict(config)}))
        result.append(row)

    for section in (article.get("abstract") or {}).get("sections", []):
        encoded = tokenizer(section["text"], add_special_tokens=False,
            return_offsets_mapping=True, truncation=False, verbose=False)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if not ids:
            continue
        if capacity <= 0 or (len(ids) > capacity and capacity <= config.overlap):
            raise WindowError(f"PMID {article['pmid']}: title leaves insufficient room for overlap")
        if len(offsets) != len(ids) or any(not 0 <= a < b <= len(section["text"]) for a, b in offsets):
            raise WindowError("Tokenizer returned invalid source offsets")
        start = 0
        while start < len(ids):
            end = min(start + capacity, len(ids))
            append(section, ids[start:end], start, end, offsets[start][0], offsets[end - 1][1])
            if end == len(ids):
                break
            start = end - config.overlap
    if not result and title_ids:
        append(None, [])
    return result


def build_windows(dataset_path: Path, tokenizer, tokenizer_info: dict, output: Path,
                  config: WindowConfig | None = None) -> dict:
    output, config = Path(output), config or WindowConfig()
    if output.exists():
        raise FileExistsError(output)
    dataset = load_dataset(dataset_path)
    rows, empty = [], []
    for article in dataset.articles:
        windows = article_windows(article, tokenizer, config)
        rows.extend(windows)
        if not windows:
            empty.append(article["pmid"])
    body = b"".join(_json(row) + b"\n" for row in rows)
    identity = {"schema_version": 1, "dataset_id": dataset.manifest["dataset_id"],
        "articles_sha256": dataset.manifest["articles_sha256"], "config": asdict(config),
        "tokenizer": tokenizer_info, "windows_sha256": _sha(body),
        "implementation_sha256": _sha(Path(__file__).read_bytes()),
        "input_format": "CLS-title_ids-section_token_slice-SEP-v1"}
    manifest = {**identity, "artifact_id": _sha(_json(identity)), "status": "complete",
        "article_count": len(dataset.articles), "window_count": len(rows),
        "empty_pmids": empty, "title_only_windows": sum(r["kind"] == "title" for r in rows),
        "max_input_token_count": max((r["input_token_count"] for r in rows), default=0)}
    output.mkdir(parents=True, exist_ok=False)
    (output / "windows.jsonl").write_bytes(body)
    (output / "manifest.json").write_bytes(_json(manifest) + b"\n")
    return manifest


def load_windows(directory: Path) -> tuple[dict, tuple[dict, ...]]:
    directory = Path(directory)
    try:
        if any((directory / name).is_symlink() for name in ("manifest.json", "windows.jsonl")):
            raise WindowError("Window files must not be symlinks")
        manifest = json.loads((directory / "manifest.json").read_bytes())
        if manifest["schema_version"] != 1 or manifest["status"] != "complete":
            raise WindowError("Incomplete or unsupported window artifact")
        body = (directory / "windows.jsonl").read_bytes()
        if _sha(body) != manifest["windows_sha256"]:
            raise WindowError("Window checksum mismatch")
        identity = {key: manifest[key] for key in ("schema_version", "dataset_id", "articles_sha256",
            "config", "tokenizer", "windows_sha256", "implementation_sha256", "input_format")}
        if _sha(_json(identity)) != manifest["artifact_id"]:
            raise WindowError("Window artifact identity mismatch")
        rows = tuple(json.loads(line) for line in body.splitlines())
        if len(rows) != manifest["window_count"] or len({r["window_id"] for r in rows}) != len(rows):
            raise WindowError("Window count or IDs mismatch")
        for row in rows:
            fields = {key: value for key, value in row.items() if key != "window_id"}
            if _sha(_json({"row": fields, "config": manifest["config"]})) != row["window_id"]:
                raise WindowError("Window identity mismatch")
            if (len(row["input_ids"]) != row["input_token_count"]
                or row["input_token_count"] > manifest["config"]["max_input_tokens"]
                or row["input_ids"][0] != 101 or row["input_ids"][-1] != 102):
                raise WindowError("Invalid model input")
        return manifest, rows
    except WindowError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise WindowError("Invalid window artifact") from exc
