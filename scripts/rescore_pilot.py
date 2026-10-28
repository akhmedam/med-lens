"""Evaluate frozen retrieval and Qwen permutations with the active pilot qrels."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from medlens.evaluation.formats import parse_queries, parse_qrels
from medlens.evaluation.metrics import evaluate, macro_average
from medlens.parsing.dataset import load_dataset
from medlens.reranking.listwise import apply_response

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("pubmed", "bm25", "dense", "rrf", "dense_qwen", "rrf_qwen")


def sha(body):
    return hashlib.sha256(body).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def rescore(profile_path, *, root=ROOT):
    profile_bytes = Path(profile_path).read_bytes()
    profile = json.loads(profile_bytes)
    if profile["schema_version"] != 1 or profile["split"] != "pilot":
        raise ValueError("Unsupported pilot profile")
    def read(name):
        entry = profile[name]
        body = (root / entry["path"]).read_bytes()
        if sha(body) != entry["sha256"]:
            raise ValueError(f"{name} checksum mismatch")
        return body
    query_bytes = read("queries")
    queries = parse_queries(query_bytes, (root / profile["queries"]["path"]).parent)
    old_qrel_bytes, qrel_bytes = read("source_qrels"), read("qrels")
    old, new = parse_qrels(old_qrel_bytes, queries), parse_qrels(qrel_bytes, queries)
    if any(set(old[q]) != set(new[q]) for q in old):
        raise ValueError("Qrels versions must cover the same pairs")
    baseline_bytes, rerank_bytes = read("frozen_retrieval"), read("frozen_reranking")
    baseline, rerank = json.loads(baseline_bytes), json.loads(rerank_bytes)
    if (baseline["queries_sha256"] != sha(query_bytes) or baseline["qrels_sha256"] != sha(old_qrel_bytes)
            or baseline["split"] != "pilot" or baseline["k"] != 10
            or rerank["queries_sha256"] != sha(query_bytes) or rerank["qrels_sha256"] != sha(old_qrel_bytes)
            or rerank["baseline_sha256"] != sha(baseline_bytes) or rerank["status"] != "complete"):
        raise ValueError("Frozen reports are not bound to the source queries/qrels")
    selected = [q for q in queries if q.split == "pilot"]
    if [q.query_id for q in selected] != [r["query_id"] for r in baseline["per_query"]]:
        raise ValueError("Frozen report query membership/order differs")
    reranks = {(r["query_id"], r["source"]): r for r in rerank["per_query"]}
    expected = {(q.query_id, m) for q in selected for m in ("dense", "rrf")}
    if len(reranks) != len(rerank["per_query"]) or set(reranks) != expected:
        raise ValueError("Reranking query/source membership differs")
    retrieval = copy.deepcopy(baseline)
    rows = []
    for query, source, target in zip(selected, baseline["per_query"], retrieval["per_query"]):
        dataset = load_dataset(query.dataset)
        if source["dataset_id"] != dataset.manifest["dataset_id"] or source["text"] != query.text:
            raise ValueError("Frozen dataset identity or query text differs")
        candidates = [a["pmid"] for a in dataset.articles]
        if not set(new[query.query_id]) <= set(candidates):
            raise ValueError("Pilot qrels contain a PMID outside the dataset")
        methods = {}
        for method in METHODS[:4]:
            ranking = [h["pmid"] for h in source["methods"][method]["ranking"]]
            before = evaluate(ranking, candidates, old[query.query_id], 10)
            if before != source["methods"][method]["metrics"]:
                raise ValueError("Source retrieval metrics do not replay")
            metrics = evaluate(ranking, candidates, new[query.query_id], 10)
            methods[method] = {"ranking": ranking, "metrics": metrics}
            target["methods"][method] = {"metrics": metrics, "ranking": [
                {"rank": i, "pmid": p, "relevance": new[query.query_id].get(p),
                 "judged": new[query.query_id].get(p) is not None} for i, p in enumerate(ranking, 1)]}
        for method in ("dense", "rrf"):
            row = reranks[(query.query_id, method)]
            original = methods[method]["ranking"]
            mapping = {f"D{i}": p for i, p in enumerate(original, 1)}
            response = apply_response(row["raw_output"], mapping,
                hit_output_limit=row["generation"]["hit_output_limit"])
            if (row["before_ranking"] != original or row["dataset_id"] != source["dataset_id"]
                    or row["prompt"]["mapping"] != mapping or row["ranking"] != response["ranking"]
                    or row["fallback"] != response["fallback"]
                    or evaluate(original, candidates, old[query.query_id], 10) != row["before"]
                    or evaluate(row["ranking"], candidates, old[query.query_id], 10) != row["after"]):
                raise ValueError("Source reranking output or metrics do not replay")
            methods[method + "_qwen"] = {"ranking": row["ranking"],
                "metrics": evaluate(row["ranking"], candidates, new[query.query_id], 10)}
        rows.append({"query_id": query.query_id, "dataset_id": source["dataset_id"], "methods": methods})
    status = "fully-judged-top-k" if all(m["metrics"]["unjudged_returned_count"] == 0
        for r in rows for m in r["methods"].values()) else "provisional-incomplete-judgments"
    identity = {"profile_sha256": sha(profile_bytes), "queries_sha256": sha(query_bytes),
        "qrels_sha256": sha(qrel_bytes), "source_retrieval_sha256": sha(baseline_bytes),
        "source_reranking_sha256": sha(rerank_bytes), "rescore_code_sha256": sha(Path(__file__).read_bytes()),
        "metric_code_sha256": sha((ROOT / "src/medlens/evaluation/metrics.py").read_bytes())}
    report = {"schema_version": 1, "run_id": sha(encoded(identity)), **identity,
        "evaluation_version": profile["evaluation_version"], "annotation": profile["annotation"],
        "split": "pilot", "k": 10, "query_count": len(rows), "comparison_status": status,
        "per_query": rows, "macro": {m: macro_average([r["methods"][m]["metrics"] for r in rows], 10)
                                     for m in METHODS}}
    retrieval.update(qrels_sha256=sha(qrel_bytes), comparison_status=status,
        rescored_from_run_id=baseline["run_id"], rescoring=identity,
        evaluation_version=profile["evaluation_version"], annotation=profile["annotation"])
    retrieval["macro"] = {m: report["macro"][m] for m in METHODS[:4]}
    retrieval["unjudged_rrf_pairs"] = [{"query_id": r["query_id"], "pmid": h["pmid"], "relevance": None}
        for r in retrieval["per_query"] for h in r["methods"]["rrf"]["ranking"] if not h["judged"]]
    retrieval["run_id"] = sha(encoded({"kind": "rescored-retrieval", **identity}))
    summary = {k: v for k, v in report.items() if k != "per_query"}
    summary["per_query"] = [{"query_id": r["query_id"], "dataset_id": r["dataset_id"],
        "metrics": {m: r["methods"][m]["metrics"] for m in METHODS}} for r in rows]
    return report, retrieval, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "benchmarks/pilot-v1/evaluation.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    paths = (args.output, args.retrieval_output, args.summary)
    if len({p.resolve() for p in paths}) != 3 or any(p.exists() for p in paths):
        parser.error("Choose three distinct, unused output paths")
    reports = rescore(args.profile)
    for path, report in zip(paths, reports):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({"version": reports[0]["evaluation_version"],
        "comparison_status": reports[0]["comparison_status"], "macro": reports[0]["macro"]}, indent=2))


if __name__ == "__main__":
    main()
