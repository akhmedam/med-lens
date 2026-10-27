"""Reciprocal rank fusion over compatible BM25 and dense candidate rankings."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

from .bm25 import BM25Config, BM25Index
from .dense import DenseIndex


class FusionError(ValueError):
    """Fusion inputs or parameters are invalid."""


def reciprocal_rank_fusion(rankings: dict[str, list[str]], candidate_order: list[str],
                           *, c: float = 60, top_k: int = 10) -> list[dict]:
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c < 0:
        raise FusionError("c must be finite and nonnegative")
    if type(top_k) is not int or top_k <= 0:
        raise FusionError("top_k must be a positive integer")
    if len(set(candidate_order)) != len(candidate_order):
        raise FusionError("Duplicate candidate PMID")
    candidates = set(candidate_order)
    positions = {}
    for method, ids in rankings.items():
        if not isinstance(method, str) or not method or len(set(ids)) != len(ids) or not set(ids) <= candidates:
            raise FusionError("Ranking contains duplicate or unknown candidates")
        positions[method] = {pmid: rank for rank, pmid in enumerate(ids, 1)}
    fused = []
    for order, pmid in enumerate(candidate_order):
        ranks = {method: ranks.get(pmid) for method, ranks in positions.items()}
        if not any(rank is not None for rank in ranks.values()):
            continue
        contributions = {method: 1 / (c + rank) if rank is not None else 0.0
                         for method, rank in ranks.items()}
        fused.append({"pmid": pmid, "score": math.fsum(contributions.values()),
            "source_ranks": ranks, "contributions": contributions, "candidate_order": order})
    fused.sort(key=lambda row: (-row["score"], row["candidate_order"]))
    result = []
    for rank, row in enumerate(fused[:top_k], 1):
        row.pop("candidate_order")
        result.append({"rank": rank, **row})
    return result


class HybridIndex:
    def __init__(self, dataset_dir: Path, windows_dir: Path, embeddings_dir: Path,
                 config: BM25Config | None = None):
        self.dense = DenseIndex(dataset_dir, windows_dir, embeddings_dir)
        self.bm25 = BM25Index(self.dense.dataset, config)

    def run(self, query: str, encoder, *, top_k: int = 10, depth: int = 30, c: float = 60) -> dict:
        if type(depth) is not int or depth <= 0 or type(top_k) is not int or not 0 < top_k <= depth:
            raise FusionError("Require positive depth and 0 < top_k <= depth")
        # Validate smoothing before expensive model inference.
        reciprocal_rank_fusion({}, [], c=c, top_k=top_k)
        lexical = self.bm25.run(query, depth)
        dense = self.dense.run(query, encoder, depth)
        if lexical["dataset_id"] != dense["dataset_id"] or lexical["query"] != dense["query"]:
            raise FusionError("Source queries or candidate datasets disagree")
        rankings = {}
        for method, run in (("bm25", lexical), ("dense", dense)):
            if len(run["results"]) > depth or [r["rank"] for r in run["results"]] != list(range(1, len(run["results"]) + 1)):
                raise FusionError("Source ranks must be contiguous and bounded by depth")
            rankings[method] = [r["pmid"] for r in run["results"]]
        started = perf_counter()
        fused = reciprocal_rank_fusion(rankings, list(self.dense.articles), c=c, top_k=top_k)
        fusion_seconds = perf_counter() - started
        dense_hits = {r["pmid"]: r for r in dense["results"]}
        lexical_hits = {r["pmid"]: r for r in lexical["results"]}
        hits = []
        for row in fused:
            article = self.dense.articles[row["pmid"]]
            d = dense_hits.get(row["pmid"])
            b = lexical_hits.get(row["pmid"])
            hits.append({**row, "title": article.get("title"),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/",
                "text_version_id": article["text_version"]["text_version_id"],
                "source_scores": {"bm25": b["score"] if b else None, "dense": d["score"] if d else None},
                "best_window": d["best_window"] if d else None})
        specification = {"dataset_id": dense["dataset_id"], "query": query, "top_k": top_k,
            "depth": depth, "c": float(c), "weights": {"bm25": 1, "dense": 1},
            "rank_origin": 1, "absent_contribution": 0, "ties": "candidate-order",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_run_ids": {"bm25": lexical["run_id"], "dense": dense["run_id"]}}
        run_id = hashlib.sha256(json.dumps(specification, sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        return {"schema_version": 1, "run_id": run_id, **specification,
            "candidate_count": len(self.dense.articles), "source_counts": {m: len(ids) for m, ids in rankings.items()},
            "fusion_seconds": fusion_seconds, "sources": {"bm25": lexical, "dense": dense}, "results": hits}
