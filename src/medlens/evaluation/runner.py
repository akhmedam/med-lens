"""Replay PubMed-order and BM25 baselines on the same frozen candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medlens.parsing.dataset import load_dataset
from medlens.retrieval.bm25 import BM25Config, BM25Index
from .formats import EvaluationError, parse_queries, parse_qrels
from .metrics import evaluate, macro_average


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def run_evaluation(queries_path: Path, qrels_path: Path, *, split: str = "pilot",
                   k: int = 10, config: BM25Config | None = None) -> dict:
    if split not in {"pilot", "dev", "test"}:
        raise EvaluationError("Choose pilot, dev, or test explicitly")
    if type(k) is not int or k <= 0:
        raise EvaluationError("k must be a positive integer")
    queries_path, qrels_path = Path(queries_path), Path(qrels_path)
    query_bytes, qrel_bytes = queries_path.read_bytes(), qrels_path.read_bytes()
    queries = parse_queries(query_bytes, queries_path.parent)
    qrels = parse_qrels(qrel_bytes, queries)
    selected = [q for q in queries if q.split == split]
    if not selected:
        raise EvaluationError(f"No queries in split: {split}")
    indexes, per_query, retrieval_ids = {}, [], []
    for query in selected:
        if query.dataset not in indexes:
            indexes[query.dataset] = BM25Index(load_dataset(query.dataset), config)
        index = indexes[query.dataset]
        candidates = [row["pmid"] for row in index.dataset.articles]
        run = index.run(query.text, k)
        retrieval_ids.append(run["run_id"])
        methods = {}
        for name, pmids in (("pubmed", candidates[:k]),
                             ("bm25", [hit["pmid"] for hit in run["results"]])):
            methods[name] = {
                "metrics": evaluate(pmids, candidates, qrels[query.query_id], k),
                "ranking": [{"rank": rank, "pmid": pmid,
                    "relevance": qrels[query.query_id].get(pmid),
                    "judged": qrels[query.query_id].get(pmid) is not None}
                    for rank, pmid in enumerate(pmids, 1)],
            }
        per_query.append({"query_id": query.query_id, "text": query.text,
            "group_id": query.group_id, "dataset_id": index.dataset.manifest["dataset_id"],
            "candidate_count": len(candidates), "bm25_run": run, "methods": methods})
    code = b"".join(name.encode() + b"\0" + Path(__file__).with_name(name).read_bytes()
                    for name in ("formats.py", "metrics.py", "runner.py"))
    specification = {"queries_sha256": _sha(query_bytes), "qrels_sha256": _sha(qrel_bytes),
        "evaluation_code_sha256": _sha(code), "split": split, "k": k,
        "query_ids": [q.query_id for q in selected], "bm25_run_ids": retrieval_ids,
        "unjudged_policy": "zero-contribution-with-coverage",
        "binary_relevance_threshold": 1, "gain": "2^rel-1",
        "recall_denominator": "all-known-relevant-qrels-including-outside-candidates"}
    run_id = _sha(json.dumps(specification, sort_keys=True, separators=(",", ":")).encode())
    return {"schema_version": 1, "run_id": run_id, **specification,
        "query_count": len(selected), "per_query": per_query,
        "macro": {name: macro_average([q["methods"][name]["metrics"] for q in per_query], k)
                  for name in ("pubmed", "bm25")}}
