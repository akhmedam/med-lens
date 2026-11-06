"""Compare completed generation summaries; no inference, tokenizer or corpus required."""

import argparse
import hashlib
import json
from pathlib import Path

from medlens.evaluation.answer_formats import compare_formats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a fresh output path")
    try:
        old, new = args.baseline.read_bytes(), args.candidate.read_bytes()
        report = compare_formats(json.loads(old), json.loads(new))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    report.update(baseline_sha256=hashlib.sha256(old).hexdigest(),
                  candidate_sha256=hashlib.sha256(new).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
