"""Create and verify a parsed dataset from an existing, complete snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medlens.ingestion.snapshots import load_snapshot
from .pubmed import PARSER_VERSION, ParseError, parse_pubmed_xml


class DatasetError(ValueError):
    """A parsed artifact is incomplete or inconsistent with its source manifest."""


@dataclass(frozen=True)
class LoadedDataset:
    manifest: dict[str, Any]
    articles: tuple[dict[str, Any], ...]


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _manifest(directory: Path, value: dict) -> None:
    temporary = directory / "manifest.json.tmp"
    temporary.write_bytes(_json(value) + b"\n")
    temporary.replace(directory / "manifest.json")


def build_dataset(snapshot_dir: Path, output: Path) -> dict[str, Any]:
    snapshot_dir, output = Path(snapshot_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_bytes = (snapshot_dir / "manifest.json").read_bytes()
    snapshot = load_snapshot(snapshot_dir)
    source = snapshot.manifest
    # Ensure the copied provenance is the same manifest that was validated.
    if json.loads(source_bytes) != source:
        raise DatasetError("Source manifest changed while it was being read")
    by_pmid = {}
    for entry in source["responses"][1:]:
        try:
            articles = parse_pubmed_xml(snapshot.responses[entry["file"]],
                source_file=entry["file"], snapshot_id=source["snapshot_id"],
                retrieved_at=entry["retrieved_at"])
        except ParseError as exc:
            raise DatasetError(f"{entry['file']}: {exc}") from exc
        for article in articles:
            if article.pmid in by_pmid:
                raise DatasetError(f"Duplicate PMID across batches: {article.pmid}")
            by_pmid[article.pmid] = article
    if set(by_pmid) != set(source["pmids"]):
        raise DatasetError("Parsed records do not match the source candidate set")
    ordered = [by_pmid[pmid] for pmid in source["pmids"]]
    body = b"".join(_json(asdict(article)) + b"\n" for article in ordered)
    code = b"".join(name.encode() + b"\0" + Path(__file__).with_name(name).read_bytes()
                    for name in ("models.py", "pubmed.py", "dataset.py"))
    manifest = {
        "schema_version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "parser_version": PARSER_VERSION, "parser_code_sha256": _sha(code),
        "source_snapshot_id": source["snapshot_id"], "source_manifest_sha256": _sha(source_bytes),
        "articles_sha256": _sha(body), "article_count": len(ordered), "pmids": source["pmids"],
        "output_file": "articles.jsonl", "source_manifest_file": "source-manifest.json",
    }
    identity = {key: manifest[key] for key in
                ("parser_code_sha256", "source_manifest_sha256", "articles_sha256")}
    manifest["dataset_id"] = _sha(_json(identity))
    output.mkdir(parents=True, exist_ok=False)
    _manifest(output, manifest)
    (output / "source-manifest.json").write_bytes(source_bytes)
    (output / "articles.jsonl").write_bytes(body)
    manifest["status"] = "complete"
    _manifest(output, manifest)
    return manifest


def load_dataset(directory: Path) -> LoadedDataset:
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_bytes())
        if (manifest["schema_version"] != 1 or manifest["status"] != "complete"
            or manifest["parser_version"] != PARSER_VERSION
            or manifest["output_file"] != "articles.jsonl"
            or manifest["source_manifest_file"] != "source-manifest.json"):
            raise DatasetError("Incomplete or unsupported parsed dataset")
        for name in ("articles.jsonl", "source-manifest.json"):
            if (directory / name).is_symlink():
                raise DatasetError("Dataset files must not be symlinks")
        body = (directory / "articles.jsonl").read_bytes()
        source_bytes = (directory / "source-manifest.json").read_bytes()
        if _sha(body) != manifest["articles_sha256"] or _sha(source_bytes) != manifest["source_manifest_sha256"]:
            raise DatasetError("Parsed dataset checksum mismatch")
        identity = {key: manifest[key] for key in
                    ("parser_code_sha256", "source_manifest_sha256", "articles_sha256")}
        if _sha(_json(identity)) != manifest["dataset_id"]:
            raise DatasetError("Dataset identity mismatch")
        source = json.loads(source_bytes)
        if (source["status"] != "complete" or source["snapshot_id"] != manifest["source_snapshot_id"]
            or source["pmids"] != manifest["pmids"]):
            raise DatasetError("Parsed dataset disagrees with the source manifest")
        rows = tuple(json.loads(line) for line in body.splitlines())
        ids = [row["pmid"] for row in rows]
        if (len(rows) != manifest["article_count"] or ids != manifest["pmids"]
            or len(set(ids)) != len(ids)):
            raise DatasetError("Parsed record count or order mismatch")
        responses = {r["file"]: r for r in source["responses"][1:]}
        for row in rows:
            provenance = row["provenance"]
            response = responses[provenance["source_file"]]
            if (provenance["snapshot_id"] != source["snapshot_id"]
                or provenance["source"] != "pubmed"
                or provenance["raw_response_sha256"] != response["sha256"]
                or provenance["retrieved_at"] != response["retrieved_at"]
                or row["text_version"]["parser_version"] != manifest["parser_version"]):
                raise DatasetError("Parsed record provenance mismatch")
        return LoadedDataset(manifest, rows)
    except DatasetError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise DatasetError("Invalid parsed dataset") from exc
