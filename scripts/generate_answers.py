"""Generate and review cited answers from a verified saved pilot context report."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from rescore_pilot import rescore
from medlens.evaluation.formats import parse_queries
from medlens.generation.context import canonical, digest, render_evidence, source_section
from medlens.generation.answers import check_context, generate_answer, render_answer
from medlens.inference.qwen import MODEL_ID, REVISION, QwenRuntime
from medlens.parsing.dataset import load_dataset
from medlens.retrieval.windows import load_windows

IDENTITY_KEYS = ("schema_version", "dataset_id", "window_artifact_id", "article_ranking", "scores_sha256",
    "config", "selection_policy", "tokenizer", "sources_sha256", "implementation_sha256")


def load_cases(path, profile_path):
    body = Path(path).read_bytes()
    artifact = json.loads(body)
    evaluation, _, _ = rescore(profile_path)
    if (artifact["profile_sha256"] != evaluation["profile_sha256"]
            or artifact["qrels_sha256"] != evaluation["qrels_sha256"]
            or artifact["evaluation_run_id"] != evaluation["run_id"]):
        raise ValueError("Context report does not belong to the current evaluation profile")
    profile = json.loads(Path(profile_path).read_bytes())
    qp = ROOT / profile["queries"]["path"]
    queries = parse_queries(qp.read_bytes(), qp.parent)
    selected = [q for q in queries if q.split == "pilot"]
    if [q.query_id for q in selected] != [r["query_id"] for r in artifact["per_query"]]:
        raise ValueError("Context query order/membership differs")
    for q, row, frozen in zip(selected, artifact["per_query"], evaluation["per_query"]):
        ctx = row["context"]
        check_context(ctx)
        if row["query"] != q.text or ctx["article_ranking"] != frozen["methods"][artifact["method"]]["ranking"]:
            raise ValueError("Context question or article ranking differs")
        dataset = load_dataset(q.dataset)
        manifest, windows = load_windows(ROOT / "data/windows/pilot-v1-002" / q.query_id)
        if (dataset.manifest["dataset_id"] != ctx["dataset_id"]
                or manifest["artifact_id"] != ctx["window_artifact_id"]):
            raise ValueError("Context dataset/window identity differs")
        articles = {a["pmid"]: a for a in dataset.articles}
        by_window = {w["window_id"]: w for w in windows}
        for source in ctx["sources"]:
            window = by_window[source["window_id"]]
            if any(source[k] != window[k] for k in ("pmid", "text_version_id", "text", "char_start",
                    "char_end", "source_path", "section_index", "provenance")):
                raise ValueError("Citation map differs from saved source window")
            article = articles[source["pmid"]]
            source_section(article, window)
            if source["title"] != article["title"] or source["section"] != window["section_label"]:
                raise ValueError("Citation title/section differs")
    return artifact, hashlib.sha256(body).hexdigest()


def control_context(parent, sources, tokenizer):
    ctx = copy.deepcopy(parent)
    ctx["sources"] = copy.deepcopy(sources)
    for i, source in enumerate(ctx["sources"], 1):
        source["citation_id"] = f"E{i}"
    ctx["article_ranking"] = list(dict.fromkeys(s["pmid"] for s in ctx["sources"]))
    ctx["sources_sha256"] = digest(ctx["sources"])
    ctx["evidence_payload"] = render_evidence(ctx["sources"])
    ctx["evidence_tokens"] = len(tokenizer.encode(ctx["evidence_payload"], add_special_tokens=False))
    ctx["selection_policy"] = "real-source-control-subset-v1"
    ctx["status"] = "source-excerpts-selected" if sources else "no-source-excerpts"
    ctx["context_id"] = digest({k: ctx[k] for k in IDENTITY_KEYS})
    return ctx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contexts", type=Path)
    parser.add_argument("--profile", type=Path, default=ROOT / "benchmarks/pilot-v1/evaluation.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--include-controls", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() == args.summary.resolve() or args.output.exists() or args.summary.exists():
        parser.error("Choose two distinct unused paths")
    artifact, artifact_hash = load_cases(args.contexts, args.profile)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, token=False,
        trust_remote_code=False, local_files_only=True)
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    cases = [{"case_id": r["query_id"], "kind": "pilot", "question": r["query"], "context": r["context"]}
             for r in artifact["per_query"]]
    for case in cases:
        info = case["context"]["tokenizer"]
        if info["model_id"] != MODEL_ID or info["revision"] != REVISION or info["backend_sha256"] != tokenizer_hash:
            raise ValueError("Generator tokenizer differs from evidence tokenizer")
        check_context(case["context"], tokenizer)
    if args.include_controls:
        parent = cases[0]["context"]
        cases.append({"case_id": "empty-context", "kind": "control", "question": cases[0]["question"],
            "context": control_context(parent, [], tokenizer), "expected": "insufficient_evidence"})
        # Real adolescent-only trial, offered for an adult comparison question.
        pediatric = [s for r in artifact["per_query"] for s in r["context"]["sources"] if s["pmid"] == "17003660"]
        if not pediatric:
            raise ValueError("Pinned wrong-population control source is absent")
        owner = next(r["context"] for r in artifact["per_query"] if pediatric[0] in r["context"]["sources"])
        cases.append({"case_id": "wrong-population", "kind": "control", "question": cases[0]["question"],
            "context": control_context(owner, [pediatric[0]], tokenizer), "expected": "insufficient_evidence"})
    identity = {"context_report_sha256": artifact_hash, "profile_sha256": artifact["profile_sha256"],
        "qrels_sha256": artifact["qrels_sha256"], "case_ids": [c["case_id"] for c in cases],
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "scripts/generate_answers.py", "src/medlens/generation/answers.py", "src/medlens/inference/qwen.py")}}
    report = {"schema_version": 1, "run_id": digest(identity), **identity, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "per_case": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:json.dump(report, f, indent=2)
    def checkpoint():
        tmp = args.output.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(args.output)
    try:
        runtime = QwenRuntime()
        report["runtime"] = runtime.metadata
        for case in cases:
            result = generate_answer(case["question"], case["context"], runtime)
            result.update(case_id=case["case_id"], kind=case["kind"], expected=case.get("expected"))
            result["rendered_answer"] = render_answer(result)
            report["per_case"].append(result)
            checkpoint()
            print(case["case_id"],result["status"],len(result["claims"]),
                round(sum(c["seconds"] for c in result["model_calls"]),2),result['reason'],flush=True)
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
    summary = {k: v for k, v in report.items() if k != "per_case"}
    summary["local_report_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary["per_case"] = [{"case_id": r["case_id"], "kind": r["kind"], "status": r["status"],
        "reason": r["reason"], "claim_count": len(r["claims"]),
        "draft_claim_count": len(r.get("draft",{}).get("claims",[])),
        "review_verdicts": [v["verdict"] for v in r["review"]],
        "stages": [{k: c[k] for k in ("stage", "seconds", "new_tokens", "hit_output_limit", "max_new_tokens")}
                   | {"input_tokens": c["prompt"]["input_tokens"], "prompt_sha256": c["prompt"]["prompt_sha256"]}
                   for c in r["model_calls"]],
        "expected": r.get("expected"), "matches_expected": r["status"] == r["expected"] if r.get("expected") else None}
        for r in report["per_case"]]
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x") as f:json.dump(summary,f,indent=2);f.write('\n')


if __name__ == "__main__":
    main()
