"""Build or inspect parsed PubMed records without network access."""

import argparse
import json
import sys
from pathlib import Path

from medlens.ingestion.snapshots import SnapshotError
from medlens.parsing.dataset import DatasetError, build_dataset, load_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("snapshot", type=Path)
    build.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "build":
            manifest = build_dataset(args.snapshot, args.output)
            directory = args.output
        else:
            manifest = load_dataset(args.directory).manifest
            directory = args.directory
    except (DatasetError, SnapshotError, OSError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"directory": str(directory), "status": manifest["status"],
        "article_count": manifest["article_count"], "dataset_id": manifest["dataset_id"],
        "parser_version": manifest["parser_version"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
