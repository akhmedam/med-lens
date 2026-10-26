"""Compare dense retrieval with existing baselines, exposing pool incompleteness."""

import hashlib
import json
from pathlib import Path

from medlens.retrieval.dense import DenseIndex
from .formats import EvaluationError, parse_queries, parse_qrels
from .metrics import evaluate, macro_average
from .runner import run_evaluation


def run_dense_evaluation(queries_path: Path, qrels_path: Path, windows_root: Path,
                         embeddings_root: Path, encoder, *, split="pilot", k=10) -> dict:
    report = run_evaluation(queries_path, qrels_path, split=split, k=k)
    query_bytes = Path(queries_path).read_bytes()
    qrel_bytes = Path(qrels_path).read_bytes()
    if (hashlib.sha256(query_bytes).hexdigest() != report["queries_sha256"]
        or hashlib.sha256(qrel_bytes).hexdigest() != report["qrels_sha256"]):
        raise EvaluationError("Evaluation inputs changed during the run")
    queries = parse_queries(query_bytes, Path(queries_path).parent)
    qrels = parse_qrels(qrel_bytes, queries)
    by_id = {q.query_id: q for q in queries}
    dense_ids, additions = [], []
    for row in report["per_query"]:
        query = by_id[row["query_id"]]
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in query.query_id):
            raise EvaluationError("Query IDs must be safe directory names for artifact roots")
        index = DenseIndex(query.dataset, Path(windows_root) / query.query_id,
            Path(embeddings_root) / query.query_id)
        run = index.run(query.text, encoder, k)
        dense_ids.append(run["run_id"])
        ids = [hit["pmid"] for hit in run["results"]]
        row["dense_run"] = run
        row["methods"]["dense"] = {"metrics": evaluate(ids, list(index.articles), qrels[query.query_id], k),
            "ranking": [{"rank": hit["rank"], "pmid": hit["pmid"],
                "relevance": qrels[query.query_id].get(hit["pmid"]),
                "judged": qrels[query.query_id].get(hit["pmid"]) is not None} for hit in run["results"]]}
        for pmid in ids:
            if qrels[query.query_id].get(pmid) is None:
                additions.append({"query_id": query.query_id, "pmid": pmid, "relevance": None})
    report["macro"]["dense"] = macro_average([r["methods"]["dense"]["metrics"] for r in report["per_query"]], k)
    identity = {"baseline_run_id": report["run_id"], "dense_run_ids": dense_ids,
        "dense_evaluation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report.update(identity)
    report["run_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report["unjudged_dense_pairs"] = additions
    report["comparison_status"] = "fully-judged-top-k" if all(
        v["metrics"]["unjudged_returned_count"] == 0 for row in report["per_query"]
        for v in row["methods"].values()) else "provisional-incomplete-judgments"
    return report
