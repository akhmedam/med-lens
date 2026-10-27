"""Evaluate all four retrieval methods on the same frozen candidate sets."""

import hashlib
import json
from pathlib import Path

from medlens.retrieval.rrf import HybridIndex
from .formats import EvaluationError, parse_queries, parse_qrels
from .metrics import evaluate, macro_average


def run_hybrid_evaluation(queries_path: Path, qrels_path: Path, windows_root: Path,
                          embeddings_root: Path, encoder, *, split="pilot", k=10, depth=30, c=60):
    query_bytes, qrel_bytes = Path(queries_path).read_bytes(), Path(qrels_path).read_bytes()
    queries = parse_queries(query_bytes, Path(queries_path).parent)
    qrels = parse_qrels(qrel_bytes, queries)
    selected = [q for q in queries if q.split == split]
    if not selected:
        raise EvaluationError("No queries in selected split")
    rows, runs, additions = [], [], []
    for query in selected:
        if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in query.query_id):
            raise EvaluationError("Query IDs must be safe directory names")
        index = HybridIndex(query.dataset, Path(windows_root) / query.query_id,
            Path(embeddings_root) / query.query_id)
        run = index.run(query.text, encoder, top_k=k, depth=depth, c=c)
        runs.append(run["run_id"])
        candidates = list(index.dense.articles)
        rankings = {"pubmed": candidates[:k],
            "bm25": [h["pmid"] for h in run["sources"]["bm25"]["results"][:k]],
            "dense": [h["pmid"] for h in run["sources"]["dense"]["results"][:k]],
            "rrf": [h["pmid"] for h in run["results"]]}
        methods = {}
        for method, ids in rankings.items():
            methods[method] = {"metrics": evaluate(ids, candidates, qrels[query.query_id], k),
                "ranking": [{"rank": rank, "pmid": pmid, "relevance": qrels[query.query_id].get(pmid),
                    "judged": qrels[query.query_id].get(pmid) is not None} for rank, pmid in enumerate(ids, 1)]}
        for pmid in rankings["rrf"]:
            if qrels[query.query_id].get(pmid) is None:
                additions.append({"query_id": query.query_id, "pmid": pmid, "relevance": None})
        rows.append({"query_id": query.query_id, "text": query.text, "group_id": query.group_id,
            "dataset_id": run["dataset_id"], "candidate_count": len(candidates), "rrf_run": run,
            "methods": methods})
    identity = {"queries_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "qrels_sha256": hashlib.sha256(qrel_bytes).hexdigest(), "query_ids": [q.query_id for q in selected],
        "split": split, "k": k, "depth": depth, "c": float(c), "rrf_run_ids": runs,
        "hybrid_evaluation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metric_code_sha256": hashlib.sha256(Path(__file__).with_name("metrics.py").read_bytes()).hexdigest(),
        "format_code_sha256": hashlib.sha256(Path(__file__).with_name("formats.py").read_bytes()).hexdigest()}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1, "run_id": run_id, **identity, "query_count": len(selected),
        "per_query": rows, "unjudged_rrf_pairs": additions,
        "comparison_status": "fully-judged-top-k" if all(v["metrics"]["unjudged_returned_count"] == 0
            for row in rows for v in row["methods"].values()) else "provisional-incomplete-judgments",
        "macro": {method: macro_average([r["methods"][method]["metrics"] for r in rows], k)
                  for method in ("pubmed", "bm25", "dense", "rrf")}}
