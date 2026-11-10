"""Replay explicitly supplied source runs; measure runtime without quality judgments."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from medlens.generation.context import canonical, locate_quote
from medlens.generation.request_contexts import verify_context_sources
from medlens.inference.backends import add_runtime_arguments, runtime_factory
from medlens.inference.qwen import REVISION
from medlens.pipeline import run_question
from medlens.reporting import render_report
from medlens.retrieval.dense import DenseIndex
from medlens.retrieval.embeddings import BGEEncoder


def sha_bytes(body):
    return hashlib.sha256(body).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def code_identity():
    names = ["scripts/benchmark_runtime.py", *["src/medlens/" + name for name in (
        "pipeline.py", "reporting.py", "generation/answers.py", "generation/context.py",
        "generation/request_contexts.py", "generation/reviews.py", "generation/contracts.py",
        "generation/quotes.py", "inference/backends.py", "inference/qwen.py", "inference/metal.py",
        "inference/mlx.py", "inference/mlx_worker.py", "retrieval/dense.py", "retrieval/embeddings.py")]]
    return {name: sha_bytes((ROOT / name).read_bytes()) for name in names}


def load_answer_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        ROOT / ".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots" / REVISION,
        local_files_only=True, token=False, trust_remote_code=False)


def release_accelerator():
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_source(path):
    """Read only this run and its explicitly recorded collection artifacts."""
    path = Path(path).resolve()
    body = path.read_bytes()
    source = json.loads(body)
    if (source.get("status") != "complete" or source.get("rerank") is not False
            or not isinstance(source.get("question"), str) or not source["question"].strip()
            or type(source.get("top_k")) is not int or not 1 <= source["top_k"] <= 10):
        raise ValueError("Require a completed primary Dense source run")
    paths = source["input_paths"]
    # Original request CLIs record absolute paths. Do not guess a different collection.
    if any(not isinstance(paths.get(key), str) or not Path(paths[key]).is_absolute()
           for key in ("dataset", "windows", "embeddings")):
        raise ValueError("Source run must declare absolute collection paths")
    dates = source.get("date_selection") or {}
    index = DenseIndex(*(Path(paths[key]) for key in ("dataset", "windows", "embeddings")),
        published_from=dates.get("published_from"), published_until=dates.get("published_until"))
    actual = {"dataset_id": index.dataset.manifest["dataset_id"],
        "window_artifact_id": index.window_manifest["artifact_id"],
        "embedding_artifact_id": index.embedding_manifest["artifact_id"]}
    if any(source.get(key) != value for key, value in actual.items()):
        raise ValueError("Source run and collection artifact identities differ")
    verify_context_sources(source["context"], index)
    ranking = [hit["pmid"] for hit in source["retrieval"]["results"]]
    if ranking != source["context"]["article_ranking"]:
        raise ValueError("Source Dense ranking and evidence ranking differ")
    return source, index, sha_bytes(body)


def input_ids_hash(call, tokenizer):
    prompt = call["prompt"]
    ids = tokenizer.apply_chat_template(prompt["messages"], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if len(ids) != prompt["input_tokens"]:
        raise ValueError("Saved prompt token count changed")
    return sha_bytes(canonical(ids).encode())


def compact_run(source, report, tokenizer, source_sha256):
    """Allowlisted, text-free summary; also usable for already saved raw replays."""
    context, original = report.get("context", {}), source["context"]
    answer = report.get("answer", {})
    previous = {call["stage"]: call for call in source.get("answer", {}).get("model_calls", [])}
    calls = []
    for call in answer.get("model_calls", []):
        actual_hash = input_ids_hash(call, tokenizer)
        if call.get("cache_reused") is not False:
            raise ValueError("Runtime measurement requires fresh model calls")
        if call.get("input_ids_sha256") != actual_hash:
            raise ValueError("Runtime and rendered input token hashes differ")
        expected_cap = {"draft": 768, "support-review": 512}.get(call["stage"])
        if expected_cap is None or call.get("max_new_tokens") != expected_cap:
            raise ValueError("Unexpected inference stage or output budget")
        before = previous.get(call["stage"])
        calls.append({"stage": call["stage"], "input_tokens": call["prompt"]["input_tokens"],
            "input_ids_sha256": actual_hash,
            "source_input_ids_sha256": input_ids_hash(before, tokenizer) if before else None,
            "same_input_ids_as_source": actual_hash == input_ids_hash(before, tokenizer) if before else None,
            **{key: call[key] for key in ("seconds", "ttft_seconds", "decode_seconds", "worker_wall_seconds",
                "new_tokens", "max_new_tokens", "hit_output_limit", "cache_reused", "peak_memory_bytes",
                "rss_peak_bytes") if key in call},
            "output_text_sha256": sha_bytes(call["text"].encode())})
    stages = [call["stage"] for call in calls]
    if stages not in ([], ["draft"], ["draft", "support-review"]):
        raise ValueError("Unexpected draft/review call order")
    runtime = report.get("runtime", {})
    fields = ("model_id", "revision", "runtime", "device", "artifact_id", "quantization", "dtype",
        "fp32_exceptions", "packages", "generation", "load_seconds", "worker_startup_seconds",
        "parameter_dtype_policy", "artifact_validation", "score_validation", "prompt_cache_reuse",
        "kv_quantization", "prefill_step_size", "ttft_scope")
    row = {"source_run_sha256": source_sha256, "request_id": report.get("request_id"),
        "source_request_id": source.get("request_id"), "status": report.get("status"),
        "answer_status": answer.get("status"), "validation_error": answer.get("validation_error"),
        "prompt_version": answer.get("prompt_version"), "review_version": answer.get("review_version"),
        "claim_count": len(answer.get("claims", [])), "model_calls": calls,
        "all_calls_fresh": all(call["cache_reused"] is False for call in calls),
        "runtime_load_count": report.get("runtime_load_count", 0),
        "runtime_closed": report.get("runtime_closed"),
        "runtime_policy": report.get("runtime_policy"),
        "runtime": {key: runtime[key] for key in fields if key in runtime},
        "implementation_sha256": report.get("implementation_sha256"),
        "identities": {key: report.get(key) for key in
            ("dataset_id", "window_artifact_id", "embedding_artifact_id")},
        "context_id": context.get("context_id"), "source_context_id": original["context_id"],
        "same_sources": bool(context) and context.get("sources") == original["sources"],
        "same_evidence_payload": bool(context) and context.get("evidence_payload") == original["evidence_payload"],
        "same_dense_ranking": [hit["pmid"] for hit in report.get("retrieval", {}).get("results", [])]
            == [hit["pmid"] for hit in source["retrieval"]["results"]],
        "timings": report.get("timings", {}),
        "source_pipeline_seconds": source.get("timings", {}).get("pipeline_seconds")}
    if report.get("error"):
        row["error"] = {key: report["error"][key] for key in ("type", "stage") if key in report["error"]}
    return row


class OwnedRuntime:
    """The public pipeline owns cleanup; the harness guarantees it after exceptions."""
    def __init__(self, instance):
        self.instance, self.closed = instance, False

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def generate(self, *args, **kwargs):
        return self.instance.generate(*args, **kwargs)

    def close(self):
        if not self.closed:
            self.instance.close()
            self.closed = True


def replay_source(path, directory, args):
    started = perf_counter()
    report = {"status": "running", "stage": "source-validation", "question": ""}
    source = tokenizer = None
    source_hash = None
    owned = []
    measurements = {}
    stages = {}
    last_stage, last_tick = None, None

    def checkpoint(value):
        nonlocal report, last_stage, last_tick
        now = perf_counter()
        stage = value["stage"]
        if last_stage is not None:
            stages[last_stage] = stages.get(last_stage, 0.0) + now - last_tick
        last_stage, last_tick = stage, now
        report = value
        write_json(directory / "run.json", value)
        (directory / "report.html").write_text(render_report(value))

    try:
        tick = perf_counter()
        source_hash = sha_bytes(Path(path).read_bytes())
        source, index, source_hash = load_source(path)
        measurements["source_validation_seconds"] = perf_counter() - tick
        report.update(question=source["question"], input_paths=source["input_paths"])
        tick = perf_counter()
        tokenizer = load_answer_tokenizer()
        measurements["answer_tokenizer_load_seconds"] = perf_counter() - tick
        base = runtime_factory(args.engine, root=ROOT, model_dir=args.mlx_model_dir,
                               worker_python=args.mlx_python)
        def make_runtime():
            runtime = OwnedRuntime(base())
            owned.append(runtime)
            return runtime
        make_runtime.identity = base.identity
        tick = perf_counter()
        report = run_question(index, source["question"],
            encoder_factory=lambda: BGEEncoder(args.model, args.tokenizer, device=args.device),
            tokenizer=tokenizer, runtime_factory=make_runtime, generate=True, rerank=False,
            top_k=source["top_k"], checkpoint=checkpoint, release_accelerator=release_accelerator)
        measurements["pipeline_wall_seconds"] = perf_counter() - tick
        report["input_paths"] = source["input_paths"]
        if "context" in report:
            verify_context_sources(report["context"], index)
            for claim in report.get("answer", {}).get("claims", []):
                for citation in claim["citations"]:
                    found = locate_quote(report["context"], citation["citation_id"], citation["quote"])
                    if any(citation.get(key) != value for key, value in found.items() if key not in {"matches", "ambiguous"}):
                        raise ValueError("Visible citation differs from exact saved source")
        if sha_bytes(Path(path).read_bytes()) != source_hash:
            raise ValueError("Source run changed during measurement")
        row = compact_run(source, report, tokenizer, source_hash)
        if report.get("status") == "complete" and not all(row[key] for key in
                ("same_sources", "same_evidence_payload", "same_dense_ranking")):
            raise ValueError("Fixed-source replay changed source selection or evidence")
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error={"type": type(exc).__name__, "message": str(exc), "stage": report.get("stage", "setup")})
        row = {"source_run_sha256": source_hash, "status": report["status"],
            "answer_status": report.get("answer", {}).get("status"),
            "error": {"type": type(exc).__name__, "stage": report.get("stage", "setup")}}
    finally:
        for runtime in owned:
            try:
                runtime.close()
            except Exception as exc:
                report.update(status="failed", cleanup_error={"type": type(exc).__name__, "message": str(exc)})
                row.update(status="failed", cleanup_error={"type": type(exc).__name__})
        if last_stage is not None:
            stages[last_stage] = stages.get(last_stage, 0.0) + perf_counter() - last_tick
    processes = [runtime.instance._process for runtime in owned if hasattr(runtime.instance, "_process")]
    row.update(**measurements, wall_seconds=perf_counter() - started, stage_wall_seconds=stages,
        owned_runtimes_closed=all(runtime.closed for runtime in owned),
        worker_reaped=all(process.poll() is not None for process in processes) if processes else None)
    if not row["owned_runtimes_closed"] or row["worker_reaped"] is False:
        report["status"] = row["status"] = "failed"
    report.update(source_run_sha256=source_hash, replay_measurements=row)
    checkpoint(report)
    row["raw_run_sha256"] = sha_bytes((directory / "run.json").read_bytes())
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, action="append", required=True,
        help="Explicit completed baseline run.json; repeat to replay more sources")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory under ignored results/")
    parser.add_argument("--summary", type=Path, required=True, help="Fresh text-free compact JSON")
    parser.add_argument("--model", type=Path, default=ROOT / ".cache/models/bge-base-en-v1.5")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / ".cache/tokenizers/bge-base-en-v1.5")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps", help="BGE device only")
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    output, summary = args.output.resolve(), args.summary.resolve()
    if (not output.is_relative_to((ROOT / "results").resolve()) or output == (ROOT / "results").resolve()
            or args.output.exists() or args.output.is_symlink()
            or args.summary.exists() or args.summary.is_symlink()
            or (summary.is_relative_to(output) and summary != output / "summary.json")):
        parser.error("Choose fresh output under results/ and a distinct fresh summary path")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "kind": "fixed-source-pipeline-replay", "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "engine": args.engine,
        "bge_device": args.device,
        "source_run_count": len(args.source_run), "semantic_quality_evaluated": False,
        "qrels_read": False, "query_sets_read": False, "heldout_discovery_performed": False,
        "scope": "Only explicit source runs and their collections; no PubMed refetch or quality scoring.",
        "timing_scope": "Each run loads tokenizer, BGE then a new Qwen runtime; wall includes source checks and cleanup, excluding the final artifact write. Stage wall includes raw checkpoint writes. Model/OS caches are not flushed.",
        "platform": platform.platform(), "code_sha256": {}, "runs": []}
    def save():
        write_json(output / "summary.json", result)
        if summary != output / "summary.json":
            write_json(summary, result)
    save()
    try:
        result["code_sha256"] = code_identity()
        for number, path in enumerate(args.source_run, 1):
            directory = output / f"run-{number:03d}"
            directory.mkdir()
            row = replay_source(path, directory, args)
            row["run_directory"] = directory.name
            result["runs"].append(row)
            save()
            print(directory.name, row["status"], row.get("answer_status"), flush=True)
        if code_identity() != result["code_sha256"]:
            raise ValueError("Measured code changed during benchmark")
        statuses = [row["status"] for row in result["runs"]]
        result["status"] = ("failed" if any(status != "complete" for status in statuses) else
            "validation_failed" if any(row.get("answer_status") == "validation_failed" for row in result["runs"]) else "complete")
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="failed", error={"type": type(exc).__name__})
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    return {"complete": 0, "validation_failed": 2}.get(result["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
