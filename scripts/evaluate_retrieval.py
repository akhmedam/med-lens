"""Evaluate saved PubMed candidates and BM25 with manual relevance judgments."""

import argparse
import json
import sys
from pathlib import Path

from medlens.evaluation.runner import run_evaluation
from medlens.parsing.dataset import DatasetError
from medlens.retrieval.bm25 import BM25Config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "dev", "test"), default="pilot")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        report = run_evaluation(args.queries, args.qrels, split=args.split, k=args.k,
            config=BM25Config(args.k1, args.b))
        body = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(body)
    except (ValueError, OSError, DatasetError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": report["run_id"], "split": report["split"],
        "query_count": report["query_count"], "macro": report["macro"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
