"""Save raw responses before validation; replay complete snapshots without HTTP."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pubmed import BASE_URL, PubMedClient, PubMedError, RawResponse, parse_search, validate_fetch


class SnapshotError(RuntimeError):
    """A snapshot is incomplete, corrupt, or inconsistent with its raw responses."""


@dataclass(frozen=True)
class LoadedSnapshot:
    manifest: dict[str, Any]
    responses: dict[str, bytes]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(directory: Path, manifest: dict) -> None:
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(directory / "manifest.json")


def download_snapshot(
    client: PubMedClient,
    query: str,
    output: Path,
    *,
    limit: int = 30,
    batch_size: int = 200,
) -> dict[str, Any]:
    if type(batch_size) is not int or not 1 <= batch_size <= 200:
        raise ValueError("batch_size must be between 1 and 200")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1, "snapshot_id": str(uuid.uuid4()),
        "status": "running", "started_at": _now(), "finished_at": None,
        "source": BASE_URL, "database": "pubmed", "query": query,
        "limit": limit, "sort": "relevance", "batch_size": batch_size,
        "client_config": asdict(client.config), "responses": [],
    }
    _write_manifest(output, manifest)

    def save(response: RawResponse, filename: str) -> None:
        (output / filename).write_bytes(response.body)
        manifest["responses"].append({
            "file": filename, "endpoint": response.endpoint,
            "parameters": response.parameters, "retrieved_at": response.retrieved_at,
            "attempts": response.attempts, "content_type": response.content_type,
            "bytes": len(response.body),
            "sha256": hashlib.sha256(response.body).hexdigest(),
        })
        _write_manifest(output, manifest)

    try:
        response = client.search(query, limit)
        save(response, "search.json")
        search = parse_search(response.body, limit)
        manifest.update(total_count=search.total_count, pmids=list(search.pmids),
                        query_translation=search.query_translation)
        _write_manifest(output, manifest)
        for offset in range(0, len(search.pmids), batch_size):
            batch = search.pmids[offset:offset + batch_size]
            response = client.fetch(batch)
            save(response, f"fetch-{offset // batch_size + 1:04d}.xml")
            validate_fetch(response.body, batch)
        manifest.update(status="complete", finished_at=_now())
        _write_manifest(output, manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", finished_at=_now(),
                        error={"type": type(exc).__name__, "message": str(exc)})
        _write_manifest(output, manifest)
        raise


def load_snapshot(directory: Path) -> LoadedSnapshot:
    """Verify saved bytes and candidate identity/order; never silently accept a partial run."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["schema_version"] != 1 or manifest["status"] != "complete":
            raise SnapshotError("Snapshot is incomplete or has an unsupported schema")
        if manifest["source"] != BASE_URL or manifest["database"] != "pubmed":
            raise SnapshotError("Unexpected snapshot source")
        entries = manifest["responses"]
        if not isinstance(entries, list) or not entries:
            raise SnapshotError("Snapshot has no search response")
        responses = {}
        for entry in entries:
            filename = entry["file"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise SnapshotError("Raw response paths must be simple filenames")
            if filename in responses:
                raise SnapshotError("Duplicate raw response file")
            path = directory / filename
            if path.is_symlink() or not path.is_file():
                raise SnapshotError(f"Missing raw response: {filename}")
            body = path.read_bytes()
            if len(body) != entry["bytes"] or hashlib.sha256(body).hexdigest() != entry["sha256"]:
                raise SnapshotError(f"Raw response checksum mismatch: {filename}")
            responses[filename] = body
        first = entries[0]
        params = first["parameters"]
        limit = manifest["limit"]
        if type(limit) is not int or not 1 <= limit <= 10_000:
            raise SnapshotError("Invalid candidate limit")
        if (first["file"] != "search.json" or first["endpoint"] != "esearch.fcgi"
            or params.get("term") != manifest["query"] or params.get("retmax") != str(limit)
            or params.get("sort") != "relevance" or params.get("db") != "pubmed"
            or params.get("retmode") != "json" or params.get("retstart") != "0"
            or manifest["sort"] != "relevance"):
            raise SnapshotError("Search parameters disagree with the snapshot")
        search = parse_search(responses["search.json"], limit)
        if (list(search.pmids) != manifest["pmids"] or search.total_count != manifest["total_count"]
            or search.query_translation != manifest["query_translation"]):
            raise SnapshotError("Search result disagrees with the saved candidate set")
        batch_size = manifest["batch_size"]
        if type(batch_size) is not int or not 1 <= batch_size <= 200:
            raise SnapshotError("Invalid batch size")
        expected_batches = [search.pmids[i:i + batch_size] for i in range(0, len(search.pmids), batch_size)]
        if len(entries) != len(expected_batches) + 1:
            raise SnapshotError("Snapshot has missing or extra batches")
        for i, (entry, batch) in enumerate(zip(entries[1:], expected_batches), start=1):
            parameters = entry["parameters"]
            if (entry["file"] != f"fetch-{i:04d}.xml" or entry["endpoint"] != "efetch.fcgi"
                or parameters.get("id") != ",".join(batch) or parameters.get("db") != "pubmed"
                or parameters.get("retmode") != "xml" or parameters.get("rettype") != "abstract"):
                raise SnapshotError("Fetch parameters disagree with the candidate batch")
            validate_fetch(responses[entry["file"]], batch)
        return LoadedSnapshot(manifest, responses)
    except SnapshotError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, PubMedError) as exc:
        raise SnapshotError("Invalid snapshot; inspect the manifest and raw responses") from exc
