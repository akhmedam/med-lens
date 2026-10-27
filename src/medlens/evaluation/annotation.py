"""Build blinded annotation packets and validate exported human judgments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medlens.parsing.dataset import load_dataset
from medlens.retrieval.bm25 import BM25Index
from .formats import EvaluationError, parse_queries, parse_qrels


def _json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def build_packet(queries_path: Path, plan_path: Path, output: Path, *, depth: int = 10,
                 dense_report: Path | None = None, qrels_path: Path | None = None,
                 supplement_method: str = "dense") -> dict:
    if type(depth) is not int or depth <= 0:
        raise EvaluationError("depth must be a positive integer")
    if supplement_method not in {"dense", "rrf"}:
        raise EvaluationError("Supplement method must be dense or rrf")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    query_bytes, plan_bytes = Path(queries_path).read_bytes(), Path(plan_path).read_bytes()
    queries = parse_queries(query_bytes, Path(queries_path).parent)
    if any(q.split != "pilot" for q in queries):
        raise EvaluationError("Annotation packet requires pilot-only queries")
    plan = json.loads(plan_bytes)
    intents = {row["query_id"]: row["relevance_intent"] for row in plan["queries"]}
    if len(intents) != len(plan["queries"]) or set(intents) != {q.query_id for q in queries}:
        raise EvaluationError("Plan and query IDs disagree")
    if (dense_report is None) != (qrels_path is None):
        raise EvaluationError("Supplement requires both dense report and base qrels")
    report, judgments, report_rows = None, None, {}
    base_hash = None
    if dense_report is not None:
        report = json.loads(Path(dense_report).read_bytes())
        qrel_bytes = Path(qrels_path).read_bytes()
        base_hash = _sha(qrel_bytes)
        if (report["queries_sha256"] != _sha(query_bytes) or report["qrels_sha256"] != base_hash
            or report["k"] != depth or report["split"] != "pilot"):
            raise EvaluationError("Dense report does not match queries, judgments, or depth")
        judgments = parse_qrels(qrel_bytes, queries)
        report_rows = {row["query_id"]: row for row in report["per_query"]}
        if set(report_rows) != {q.query_id for q in queries}:
            raise EvaluationError("Dense report must cover all pilot queries")
    items, sources = [], []
    for query in queries:
        dataset = load_dataset(query.dataset)
        by_id = {row["pmid"]: row for row in dataset.articles}
        run = BM25Index(dataset).run(query.text, depth)
        methods = {"pubmed": list(by_id)[:depth], "bm25": [r["pmid"] for r in run["results"]]}
        pooled = set(methods["pubmed"]) | set(methods["bm25"])
        dense_id = None
        if report is not None:
            dense = report_rows[query.query_id][supplement_method + "_run"]
            if dense["dataset_id"] != dataset.manifest["dataset_id"] or dense["query"] != query.text:
                raise EvaluationError("Dense report candidate identity or query changed")
            methods[supplement_method] = [hit["pmid"] for hit in dense["results"][:depth]]
            if len(set(methods[supplement_method])) != len(methods[supplement_method]) or not set(methods[supplement_method]) <= set(by_id):
                raise EvaluationError("Invalid dense candidate ranking")
            pooled = {pmid for pmid in methods[supplement_method] if judgments[query.query_id].get(pmid) is None}
            dense_id = dense["run_id"]
        order = sorted(pooled, key=lambda pmid: _sha(f"pilot-blind-v1:{query.query_id}:{pmid}".encode()))
        for pmid in order:
            row = by_id[pmid]
            items.append({"query_id": query.query_id, "query": query.text,
                "intent": intents[query.query_id], "pmid": pmid, "title": row["title"],
                "abstract": row["abstract"], "abstract_status": row["abstract_status"],
                "text_version_id": row["text_version"]["text_version_id"]})
        sources.append({"query_id": query.query_id, "dataset_id": dataset.manifest["dataset_id"],
            "bm25_run_id": run["run_id"], "supplement_method": supplement_method if report else None,
            "supplement_run_id": dense_id, "methods": methods, "pool_count": len(order)})
    if not items:
        raise EvaluationError("No candidates available for annotation")
    template = Path(__file__).with_name("annotation.html").read_text()
    identity = {"schema_version": 1, "queries_sha256": _sha(query_bytes),
        "plan_sha256": _sha(plan_bytes), "depth": depth, "sources": sources,
        "items_sha256": _sha(_json(items)), "builder_sha256": _sha(Path(__file__).read_bytes()),
        "template_sha256": _sha(template.encode())}
    if report is not None:
        identity.update(supplement_to_qrels_sha256=base_hash, retrieval_evaluation_run_id=report["run_id"],
            supplement_method=supplement_method)
    pool_id = _sha(_json(identity))
    packet = {"schema_version": 1, "pool_id": pool_id, "items": items}
    # Source text is untrusted: it must not be able to close the JSON script tag.
    embedded = _json(packet).decode().replace("<", "\\u003c").replace("&", "\\u0026")
    html = template.replace("__PACKET_JSON__", embedded)
    output.mkdir(parents=True, exist_ok=False)
    (output / "pool.json").write_bytes(_json(packet) + b"\n")
    (output / "index.html").write_text(html, encoding="utf-8")
    manifest = {**identity, "pool_id": pool_id, "item_count": len(items),
        "query_count": len(queries), "pool_sha256": _sha(_json(packet) + b"\n"),
        "html_sha256": _sha(html.encode()), "status": "awaiting-human-judgments"}
    (output / "manifest.json").write_bytes(_json(manifest) + b"\n")
    return manifest


def export_qrels(pool_dir: Path, annotations_path: Path, output: Path,
                 *, allow_partial: bool = False) -> dict:
    pool_dir, output = Path(pool_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    receipt_path = output.with_name(output.name + ".receipt.json")
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    pool_bytes = (pool_dir / "pool.json").read_bytes()
    manifest = json.loads((pool_dir / "manifest.json").read_bytes())
    if _sha(pool_bytes) != manifest["pool_sha256"]:
        raise EvaluationError("Pool checksum mismatch")
    pool = json.loads(pool_bytes)
    annotation_bytes = Path(annotations_path).read_bytes()
    labels = json.loads(annotation_bytes)
    if (labels.get("schema_version") != 1 or labels.get("pool_id") != pool["pool_id"]
        or pool["pool_id"] != manifest["pool_id"]):
        raise EvaluationError("Annotations belong to a different pool")
    rows = labels.get("judgments")
    if not isinstance(rows, list):
        raise EvaluationError("Expected a judgments list")
    # Reuse the evaluation grade/duplicate validation, with only pool query IDs.
    query_ids = dict.fromkeys(item["query_id"] for item in pool["items"])
    dummy = b"".join(_json({"query_id": qid, "text": "annotation", "split": "pilot",
        "group_id": qid, "dataset": "."}) + b"\n" for qid in query_ids)
    parsed = parse_qrels(b"".join(_json(row) + b"\n" for row in rows),
        parse_queries(dummy, pool_dir))
    expected = {(item["query_id"], item["pmid"]) for item in pool["items"]}
    actual = {(qid, pmid) for qid, grades in parsed.items() for pmid in grades}
    if actual != expected:
        raise EvaluationError("Annotations must contain exactly the pool query/PMID pairs")
    judged = sum(grade is not None for grades in parsed.values() for grade in grades.values())
    if judged != len(expected) and not allow_partial:
        raise EvaluationError("Incomplete annotations; use --allow-partial for an interim report")
    body = b"".join(_json(row) + b"\n" for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(body)
    receipt = {"schema_version": 1, "pool_id": pool["pool_id"],
        "annotations_sha256": _sha(annotation_bytes), "qrels_sha256": _sha(body),
        "judged_count": judged, "pool_count": len(expected), "complete": judged == len(expected)}
    # Keep pool binding next to the standard evaluator-compatible qrels.
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def merge_supplement(queries_path: Path, base_path: Path, supplement_path: Path,
                     pool_dir: Path, output: Path) -> dict:
    output, pool_dir = Path(output), Path(pool_dir)
    receipt_path = output.with_name(output.name + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError(output)
    base_bytes, supplement_bytes = Path(base_path).read_bytes(), Path(supplement_path).read_bytes()
    pool_bytes = (pool_dir / "pool.json").read_bytes()
    manifest = json.loads((pool_dir / "manifest.json").read_bytes())
    receipt = json.loads(Path(str(supplement_path) + ".receipt.json").read_bytes())
    if (manifest.get("supplement_to_qrels_sha256") != _sha(base_bytes)
        or manifest["pool_sha256"] != _sha(pool_bytes)
        or receipt["pool_id"] != manifest["pool_id"]
        or receipt["qrels_sha256"] != _sha(supplement_bytes) or not receipt["complete"]):
        raise EvaluationError("Supplement binding or completion check failed")
    query_bytes = Path(queries_path).read_bytes()
    if manifest["queries_sha256"] != _sha(query_bytes):
        raise EvaluationError("Supplement query set changed")
    queries = parse_queries(query_bytes, Path(queries_path).parent)
    base, extra = parse_qrels(base_bytes, queries), parse_qrels(supplement_bytes, queries)
    expected = {(row["query_id"], row["pmid"]) for row in json.loads(pool_bytes)["items"]}
    if {(qid, pmid) for qid, grades in extra.items() for pmid in grades} != expected:
        raise EvaluationError("Supplement pair set mismatch")
    for qid, grades in extra.items():
        for pmid, grade in grades.items():
            if grade is None or (base[qid].get(pmid) is not None and base[qid][pmid] != grade):
                raise EvaluationError("Missing or conflicting supplemental judgment")
            base[qid][pmid] = grade
    body = b"".join(_json({"query_id": qid, "pmid": pmid, "relevance": grade}) + b"\n"
                    for qid, grades in base.items() for pmid, grade in grades.items())
    result = {"qrels_sha256": _sha(body), "base_qrels_sha256": _sha(base_bytes),
        "supplement_qrels_sha256": _sha(supplement_bytes), "supplement_pool_id": manifest["pool_id"],
        "complete": all(grade is not None for grades in base.values() for grade in grades.values()),
        "judgment_count": sum(len(grades) for grades in base.values())}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(body)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result
