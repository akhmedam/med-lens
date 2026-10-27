"""Build a blinded pilot packet or validate and export its human judgments."""

import argparse
import json
import sys
from pathlib import Path

from medlens.evaluation.annotation import build_packet, export_qrels, merge_supplement
from medlens.parsing.dataset import DatasetError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("queries", type=Path)
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--depth", type=int, default=10)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--dense-report", "--retrieval-report", dest="dense_report", type=Path)
    build.add_argument("--method", choices=("dense", "rrf"), default="dense")
    build.add_argument("--qrels", type=Path)
    export = commands.add_parser("export")
    export.add_argument("pool", type=Path)
    export.add_argument("--annotations", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--allow-partial", action="store_true")
    merge = commands.add_parser("merge")
    merge.add_argument("queries", type=Path)
    merge.add_argument("--base", type=Path, required=True)
    merge.add_argument("--supplement", type=Path, required=True)
    merge.add_argument("--pool", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build_packet(args.queries, args.plan, args.output, depth=args.depth,
                dense_report=args.dense_report, qrels_path=args.qrels, supplement_method=args.method)
            result = {key: result[key] for key in ("pool_id", "item_count", "query_count", "status")}
        elif args.command == "export":
            result = export_qrels(args.pool, args.annotations, args.output, allow_partial=args.allow_partial)
        else:
            result = merge_supplement(args.queries, args.base, args.supplement, args.pool, args.output)
    except (OSError, ValueError, KeyError, TypeError, DatasetError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
