"""Search a frozen PubMed candidate dataset using exact BGE cosine similarity."""

import argparse
import json
import sys
from pathlib import Path

from medlens.retrieval.dense import DenseIndex
from medlens.retrieval.embeddings import BGEEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", type=Path, default=Path(".cache/models/bge-base-en-v1.5"))
    parser.add_argument("--tokenizer", type=Path, default=Path(".cache/tokenizers/bge-base-en-v1.5"))
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(args.output)
        index = DenseIndex(args.dataset, args.windows, args.embeddings)
        encoder = BGEEncoder(args.model, args.tokenizer, device=args.device)
        report = index.run(args.query, encoder, args.top_k)
        body = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(body)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
