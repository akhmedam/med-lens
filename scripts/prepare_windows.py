"""Download pinned tokenizer assets, build windows offline, or verify an artifact."""

import argparse
import json
import sys
from pathlib import Path

from medlens.parsing.dataset import DatasetError
from medlens.retrieval.bge_tokenizer import download_tokenizer, load_tokenizer
from medlens.retrieval.windows import WindowConfig, build_windows, load_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tokenizer = commands.add_parser("tokenizer")
    tokenizer.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("dataset", type=Path)
    build.add_argument("--tokenizer", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--body-tokens", type=int, default=384)
    build.add_argument("--overlap", type=int, default=64)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "tokenizer":
            result = download_tokenizer(args.output)
        elif args.command == "build":
            tokenizer, info = load_tokenizer(args.tokenizer)
            manifest = build_windows(args.dataset, tokenizer, info, args.output,
                WindowConfig(args.body_tokens, args.overlap))
            result = {key: manifest[key] for key in ("artifact_id", "article_count", "window_count", "empty_pmids", "max_input_token_count")}
        else:
            result = load_windows(args.directory)[0]
    except (OSError, ValueError, KeyError, TypeError, DatasetError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
