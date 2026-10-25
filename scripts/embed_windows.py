"""Download pinned BGE weights, embed saved windows offline, or inspect vectors."""

import argparse
import json
import sys
from pathlib import Path

from medlens.retrieval.embeddings import BGEEncoder, build_embeddings, download_model, load_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model")
    model.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("windows", type=Path)
    build.add_argument("--model", type=Path, required=True)
    build.add_argument("--tokenizer", type=Path, required=True)
    build.add_argument("--device", choices=("mps", "cpu"), default="mps")
    build.add_argument("--batch-size", type=int, default=16)
    build.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "model":
            result = download_model(args.output)
        elif args.command == "build":
            if args.output.exists():
                raise FileExistsError(args.output)
            encoder = BGEEncoder(args.model, args.tokenizer, device=args.device, batch_size=args.batch_size)
            result = build_embeddings(args.windows, encoder, args.output)
        else:
            result = load_embeddings(args.directory)[0]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
