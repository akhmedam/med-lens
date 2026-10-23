"""Build a blinded pilot packet or validate and export its human judgments."""

import argparse
import json
import sys
from pathlib import Path

from medlens.evaluation.annotation import build_packet, export_qrels
from medlens.parsing.dataset import DatasetError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("queries", type=Path)
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--depth", type=int, default=10)
    build.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("pool", type=Path)
    export.add_argument("--annotations", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build_packet(args.queries, args.plan, args.output, depth=args.depth)
            result = {key: result[key] for key in ("pool_id", "item_count", "query_count", "status")}
        else:
            result = export_qrels(args.pool, args.annotations, args.output, allow_partial=args.allow_partial)
    except (OSError, ValueError, KeyError, TypeError, DatasetError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
