"""Fetch or verify each pilot snapshot, then build its parsed candidate dataset."""

import argparse
import json
import sys
from pathlib import Path

from medlens.evaluation.formats import EvaluationError, parse_queries
from medlens.ingestion.pubmed import PubMedClient, PubMedError
from medlens.ingestion.snapshots import download_snapshot, load_snapshot, SnapshotError
from medlens.parsing.dataset import build_dataset, load_dataset, DatasetError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="Allow network calls for missing snapshots")
    args = parser.parse_args()
    try:
        queries = parse_queries(args.queries.read_bytes(), args.queries.parent)
        plan = json.loads(args.plan.read_bytes())
        entries = {row["query_id"]: row for row in plan["queries"]}
        if (len(entries) != len(plan["queries"]) or set(entries) != {q.query_id for q in queries}
            or any(q.split != "pilot" for q in queries)):
            raise EvaluationError("Plan must match a pilot-only query set")
        limit = plan["candidate_limit"]
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise EvaluationError("Invalid candidate limit")
        # Validate directory names before any writes or network calls.
        if any(not q.query_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in q.query_id) for q in queries):
            raise EvaluationError("Pilot query IDs must be safe directory names")
        client = PubMedClient()
        for query in queries:
            entry = entries[query.query_id]
            source = args.snapshots / query.query_id
            if not source.exists() and args.fetch:
                download_snapshot(client, entry["pubmed_query"], source, limit=limit)
            snapshot = load_snapshot(source).manifest
            if snapshot["query"] != entry["pubmed_query"] or snapshot["limit"] != limit:
                raise EvaluationError("Saved snapshot disagrees with the frozen plan")
            if not query.dataset.exists():
                build_dataset(source, query.dataset)
            dataset = load_dataset(query.dataset).manifest
            if dataset["source_snapshot_id"] != snapshot["snapshot_id"]:
                raise EvaluationError("Parsed dataset belongs to a different snapshot")
            print(json.dumps({"query_id": query.query_id, "candidate_count": dataset["article_count"],
                "dataset_id": dataset["dataset_id"]}), flush=True)
    except (OSError, ValueError, KeyError, TypeError, PubMedError, SnapshotError, DatasetError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
