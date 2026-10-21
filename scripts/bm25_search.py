"""Search one verified parsed dataset with BM25; no network or model download."""

import argparse
import json
import sys
from pathlib import Path

from medlens.parsing.dataset import DatasetError, load_dataset
from medlens.retrieval.bm25 import BM25Config, BM25Index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--output", type=Path, help="Save a JSON run to a new file")
    args = parser.parse_args()
    try:
        if args.output and args.output.exists():
            raise FileExistsError(args.output)
        report = BM25Index(load_dataset(args.dataset), BM25Config(args.k1, args.b)).run(
            args.query, args.top_k)
        body = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(body)
    except (DatasetError, OSError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
