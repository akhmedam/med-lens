"""Download a PubMed snapshot or inspect a saved one without network access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medlens.ingestion.pubmed import ClientConfig, PubMedClient, PubMedError
from medlens.ingestion.snapshots import SnapshotError, download_snapshot, load_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--query", required=True)
    fetch.add_argument("--limit", type=int, default=30)
    fetch.add_argument("--batch-size", type=int, default=200)
    fetch.add_argument("--timeout", type=float, default=20)
    fetch.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            manifest = download_snapshot(PubMedClient(ClientConfig(timeout=args.timeout)),
                args.query, args.output, limit=args.limit, batch_size=args.batch_size)
            directory = args.output
        else:
            manifest = load_snapshot(args.directory).manifest
            directory = args.directory
    except (PubMedError, SnapshotError, OSError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"directory": str(directory), "status": manifest["status"],
        "total_matches": manifest["total_count"], "selected_count": len(manifest["pmids"]),
        "pmids": manifest["pmids"], "raw_files": len(manifest["responses"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
