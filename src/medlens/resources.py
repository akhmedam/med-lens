"""Portable identities for rebuilt development collections, separate from history."""

import hashlib
import json
from pathlib import Path
import re

from medlens.evaluation.formats import parse_queries


def _relative(root, name, prefix):
    if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
            or not name.startswith(prefix)):
        raise ValueError("Unsupported resource path: " + str(name))
    path = (root/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Resource path escapes checkout")
    return path


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_collection_profile(root, path):
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root/path
    body = path.read_bytes()
    profile = json.loads(body)
    if (not isinstance(profile, dict) or set(profile) != {"schema_version", "kind", "version",
            "historical_snapshot_restore", "queries", "collections"}
            or type(profile["schema_version"]) is not int or profile["schema_version"] != 1
            or profile["kind"] != "rebuilt-pilot" or profile["historical_snapshot_restore"] is not False
            or not isinstance(profile["version"], str) or not profile["version"].strip()):
        raise ValueError("Require a versioned rebuilt-pilot profile")
    query_entry = profile["queries"]
    if not isinstance(query_entry, dict) or set(query_entry) != {"path", "sha256"}:
        raise ValueError("Invalid query identity")
    query_path = _relative(root, query_entry["path"], "benchmarks/pilot-v1/")
    query_bytes = query_path.read_bytes()
    if hashlib.sha256(query_bytes).hexdigest() != query_entry["sha256"]:
        raise ValueError("Pilot query identity differs")
    queries = parse_queries(query_bytes, query_path.parent)
    if any(q.split != "pilot" for q in queries):
        raise ValueError("Only development pilot queries are allowed")
    rows = profile["collections"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Collection list is empty")
    fields = {"query_id", "dataset", "windows", "embeddings", "dataset_id",
              "window_artifact_id", "embedding_artifact_id"}
    resolved = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("Unexpected collection fields")
        if not all(_digest(row[key]) for key in ("dataset_id", "window_artifact_id", "embedding_artifact_id")):
            raise ValueError("Invalid collection identity")
        resolved.append({**row, **{key: _relative(root, row[key], "data/")
                                 for key in ("dataset", "windows", "embeddings")}})
    if [r["query_id"] for r in rows] != [q.query_id for q in queries]:
        raise ValueError("Collections must match pilot query membership and order")
    return profile, hashlib.sha256(body).hexdigest(), resolved
