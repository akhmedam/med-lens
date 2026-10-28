"""Rerank frozen dense and RRF top-10 lists with one local Qwen instance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from medlens.evaluation.formats import parse_queries, parse_qrels
from medlens.evaluation.metrics import evaluate, macro_average
from medlens.parsing.dataset import load_dataset
from medlens.reranking.listwise import build_prompt, apply_response
from medlens.reranking.qwen import QwenReranker


def sha(body):
    return hashlib.sha256(body).hexdigest()


def prepare(queries_path, qrels_path, baseline_path):
    query_bytes, qrel_bytes = queries_path.read_bytes(), qrels_path.read_bytes()
    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes)
    if baseline["queries_sha256"] != sha(query_bytes) or baseline["qrels_sha256"] != sha(qrel_bytes):
        raise ValueError("Baseline queries/qrels hashes do not match")
    if baseline["k"] != 10 or baseline["comparison_status"] not in {
            "fully-judged-top-k", "provisional-incomplete-judgments"}:
        raise ValueError("Require a supported top-10 baseline")
    queries = parse_queries(query_bytes, queries_path.parent)
    qrels = parse_qrels(qrel_bytes, queries)
    selected = [q for q in queries if q.split == baseline["split"]]
    if [q.query_id for q in selected] != [r["query_id"] for r in baseline["per_query"]]:
        raise ValueError("Baseline query order/membership differs")
    jobs = []
    for query, row in zip(selected, baseline["per_query"]):
        dataset = load_dataset(query.dataset)
        if dataset.manifest["dataset_id"] != row["dataset_id"] or query.text != row["text"]:
            raise ValueError("Baseline dataset or query text differs")
        articles = {a["pmid"]: a for a in dataset.articles}
        for method in ("dense", "rrf"):
            ranking = [h["pmid"] for h in row["methods"][method]["ranking"]]
            metrics = evaluate(ranking, list(articles), qrels[query.query_id], 10)
            if metrics != row["methods"][method]["metrics"] or len(ranking) != 10:
                raise ValueError("Baseline metrics or top-10 size differs")
            jobs.append((query, method, ranking, articles, metrics, dataset.manifest["dataset_id"]))
    return jobs, qrels, {"baseline_sha256": sha(baseline_bytes), "baseline_run_id": baseline["run_id"],
        "queries_sha256": sha(query_bytes), "qrels_sha256": sha(qrel_bytes), "split": baseline["split"],
        "comparison_status": baseline["comparison_status"],
        "annotation": baseline.get("annotation"), "evaluation_version": baseline.get("evaluation_version")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a fresh path")
    jobs, qrels, identity = prepare(args.queries, args.qrels, args.baseline)
    report = {"schema_version": 1, "status": "running", **identity,
        "started_at": datetime.now(timezone.utc).isoformat(), "per_query": [],
        "code_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in
            (Path(__file__), ROOT / "src/medlens/reranking/listwise.py",
             ROOT / "src/medlens/reranking/qwen.py", ROOT / "src/medlens/evaluation/metrics.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path before loading the model. Later checkpoints replace only this run.
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    def checkpoint():
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(args.output)
    try:
        model = QwenReranker()
        report["runtime"] = model.metadata
        for query, method, ranking, articles, metrics, dataset_id in jobs:
            prompt = build_prompt(query.text, [articles[p] for p in ranking], model.tokenizer)
            generated = model.generate(prompt)
            result = apply_response(generated["text"], prompt["mapping"],
                                    hit_output_limit=generated["hit_output_limit"])
            after = evaluate(result["ranking"], list(articles), qrels[query.query_id], 10)
            report["per_query"].append({"query_id": query.query_id, "source": method,
                "dataset_id": dataset_id, "before_ranking": ranking, **result,
                "prompt": prompt, "generation": generated, "before": metrics, "after": after})
            checkpoint()
            print(f'{query.query_id} {method}: {generated["seconds"]:.2f}s; '
                  f'fallback={result["fallback"]}; nDCG {metrics["ndcg@10"]:.4f} -> {after["ndcg@10"]:.4f}', flush=True)
        report["macro"] = {method: {stage: macro_average([r[stage] for r in report["per_query"]
            if r["source"] == method], 10) for stage in ("before", "after")}
            for method in ("dense", "rrf")}
        report["fallback_count"] = sum(r["fallback"] for r in report["per_query"])
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
        checkpoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
