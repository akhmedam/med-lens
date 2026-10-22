"""Strict JSONL inputs for query sets and article relevance judgments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class EvaluationError(ValueError):
    """Evaluation inputs are inconsistent or unsupported."""


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    split: str
    group_id: str
    dataset: Path


def _rows(body: bytes, name: str):
    try:
        for number, line in enumerate(body.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected an object")
            yield number, row
    except (ValueError, UnicodeError) as exc:
        raise EvaluationError(f"Invalid JSONL in {name}: {exc}") from exc


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{field} must be a nonempty string")
    return value


def parse_queries(body: bytes, base: Path) -> tuple[Query, ...]:
    queries, seen, groups = [], set(), {}
    for number, row in _rows(body, "queries"):
        if set(row) != {"query_id", "text", "split", "group_id", "dataset"}:
            raise EvaluationError(f"Unexpected query fields at line {number}")
        values = {key: _text(value, key) for key, value in row.items()}
        qid, split, group = values["query_id"], values["split"], values["group_id"]
        if split not in {"pilot", "dev", "test"}:
            raise EvaluationError(f"Unsupported split: {split}")
        if qid in seen:
            raise EvaluationError(f"Duplicate query: {qid}")
        if group in groups and groups[group] != split:
            raise EvaluationError(f"Paraphrase group crosses splits: {group}")
        seen.add(qid)
        groups[group] = split
        queries.append(Query(qid, values["text"], split, group,
            (base / values["dataset"]).resolve()))
    if not queries:
        raise EvaluationError("Query set is empty")
    return tuple(queries)


def parse_qrels(body: bytes, queries: tuple[Query, ...]) -> dict[str, dict[str, int | None]]:
    qrels = {q.query_id: {} for q in queries}
    for number, row in _rows(body, "qrels"):
        if set(row) != {"query_id", "pmid", "relevance"}:
            raise EvaluationError(f"Unexpected judgment fields at line {number}")
        qid, pmid = _text(row["query_id"], "query_id"), _text(row["pmid"], "pmid")
        if not pmid.isascii() or not pmid.isdecimal() or int(pmid) <= 0:
            raise EvaluationError(f"Invalid PMID: {pmid}")
        grade = row["relevance"]
        if grade is not None and (type(grade) is not int or grade not in {0, 1, 2}):
            raise EvaluationError("Relevance must be 0, 1, 2, or null")
        if qid not in qrels:
            raise EvaluationError(f"Unknown query in judgments: {qid}")
        if pmid in qrels[qid]:
            raise EvaluationError(f"Duplicate judgment: {qid}/{pmid}")
        qrels[qid][pmid] = grade
    return qrels
