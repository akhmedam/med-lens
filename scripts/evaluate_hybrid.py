"""Compare PubMed, BM25, dense, and RRF on frozen datasets and common qrels."""

import argparse
import json
import sys
from pathlib import Path

from medlens.evaluation.hybrid_evaluation import run_hybrid_evaluation
from medlens.retrieval.embeddings import BGEEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--windows-root", type=Path, required=True)
    parser.add_argument("--embeddings-root", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "dev", "test"), default="pilot")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--c", type=float, default=60)
    parser.add_argument("--model", type=Path, default=Path(".cache/models/bge-base-en-v1.5"))
    parser.add_argument("--tokenizer", type=Path, default=Path(".cache/tokenizers/bge-base-en-v1.5"))
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        encoder = BGEEncoder(args.model, args.tokenizer, device=args.device)
        report = run_hybrid_evaluation(args.queries, args.qrels, args.windows_root,
            args.embeddings_root, encoder, split=args.split, k=args.k, depth=args.depth, c=args.c)
        body = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(body)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": report["run_id"], "comparison_status": report["comparison_status"],
        "new_judgments_needed": len(report["unjudged_rrf_pairs"]), "macro": report["macro"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
