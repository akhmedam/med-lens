"""Graded ranking metrics with explicit denominators and missing judgments."""

from __future__ import annotations

import math

from .formats import EvaluationError


def evaluate(ranking: list[str], candidates: list[str], qrels: dict[str, int | None],
             k: int = 10) -> dict:
    if type(k) is not int or k <= 0:
        raise EvaluationError("k must be a positive integer")
    if len(set(ranking)) != len(ranking) or len(set(candidates)) != len(candidates):
        raise EvaluationError("Duplicate PMID in ranking or candidates")
    if not set(ranking) <= set(candidates):
        raise EvaluationError("Ranking contains a PMID outside its candidate set")
    if any(grade is not None and (type(grade) is not int or grade not in {0, 1, 2})
           for grade in qrels.values()):
        raise EvaluationError("Invalid relevance grade")
    judged = {pmid: grade for pmid, grade in qrels.items() if grade is not None}
    relevant = {pmid for pmid, grade in judged.items() if grade > 0}
    top = ranking[:k]
    hits = [i for i, pmid in enumerate(top, 1) if pmid in relevant]
    dcg = sum((2 ** judged.get(pmid, 0) - 1) / math.log2(i + 1)
              for i, pmid in enumerate(top, 1))
    ideal = sorted(judged.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(i + 1) for i, grade in enumerate(ideal, 1))
    # Unjudged contributes zero to pooled metrics, but never becomes a grade 0.
    return {
        f"ndcg@{k}": dcg / idcg if idcg else None,
        f"precision@{k}": len(hits) / k if judged else None,
        f"recall@{k}": len(hits) / len(relevant) if relevant else None,
        f"mrr@{k}": (1 / hits[0] if hits else 0.0) if relevant else None,
        f"judged_coverage@{k}": sum(pmid in judged for pmid in top) / len(top) if top else None,
        "candidate_coverage": len(relevant & set(candidates)) / len(relevant) if relevant else None,
        "candidate_judged_coverage": sum(pmid in judged for pmid in candidates) / len(candidates) if candidates else None,
        "returned_count": len(top), "judged_returned_count": sum(pmid in judged for pmid in top),
        "unjudged_returned_count": sum(pmid not in judged for pmid in top),
        "judged_count": len(judged), "known_relevant_count": len(relevant),
        "known_relevant_in_candidates": len(relevant & set(candidates)),
    }


def macro_average(rows: list[dict], k: int) -> dict:
    keys = [f"ndcg@{k}", f"precision@{k}", f"recall@{k}", f"mrr@{k}",
            f"judged_coverage@{k}", "candidate_coverage", "candidate_judged_coverage"]
    result = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = {"mean": sum(values) / len(values) if values else None,
                       "defined_queries": len(values), "total_queries": len(rows)}
    return result
