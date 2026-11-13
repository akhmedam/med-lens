"""Lazy local API adapter for the existing PubMed -> BGE -> Qwen workflow.

The service owns its job directory and passes a fresh child path to this runner.
Raw error messages belong to private artifacts; HTTP responses must use the
service's sanitized projection.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace


def _load_pipeline():
    """Import the existing implementations only when a queued job starts."""
    from medlens.inference.profiles import load_answer_tokenizer
    from medlens.inference.backends import runtime_factory
    from medlens.live import local_encoder_policy, run_live_question
    from medlens.ingestion.pubmed import PubMedClient
    from medlens.retrieval.bge_tokenizer import load_tokenizer
    from medlens.retrieval.embeddings import BGEEncoder
    return SimpleNamespace(load_answer_tokenizer=load_answer_tokenizer,
        runtime_factory=runtime_factory, local_encoder_policy=local_encoder_policy,
        run_live_question=run_live_question, load_tokenizer=load_tokenizer,
        BGEEncoder=BGEEncoder, PubMedClient=PubMedClient)


def _release_mps():
    # run_live_question already collects the phase's objects before this hook.
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _artifact_error(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def _persist_setup_report(output, report):
    """Best-effort private artifacts, without claiming an existing directory."""
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        report["artifact_error"] = _artifact_error(exc)
        return
    try:
        from medlens.reporting import render_report
        with (output / "report.html").open("x", encoding="utf-8") as stream:
            stream.write(render_report(report))
    except Exception as exc:
        report["artifact_error"] = _artifact_error(exc)
    try:
        with (output / "run.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except OSError as exc:
        report["artifact_error"] = _artifact_error(exc)


def make_local_runner(root, cache_root):
    """Return a per-job adapter without importing ML code or loading assets.

    Each call creates a new native runtime factory. The existing live pipeline
    controls when to instantiate and close that runtime and retains its original
    report, including cleanup failures. One lazy PubMed client keeps its limiter
    across serialized jobs; no resident model or fallback is added.
    """
    root, cache_root = Path(root).resolve(), Path(cache_root).resolve()
    client = None

    def run(request, output, progress=None):
        nonlocal client
        output = Path(output)
        # A service bug or reused path must not overwrite an earlier job or link.
        if output.exists() or output.is_symlink():
            raise FileExistsError("Choose a fresh output directory")
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            # Match ask_live.py before any Transformers/MLX-related import.
            os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            pipeline = _load_pipeline()
            make_runtime = pipeline.runtime_factory("mlx", model="4b", root=root)
            model_dir = root / ".cache/models/bge-base-en-v1.5"
            tokenizer_dir = root / ".cache/tokenizers/bge-base-en-v1.5"
            policy = pipeline.local_encoder_policy(model_dir, device="mps")
            if client is None:
                client = pipeline.PubMedClient()
        except Exception as exc:
            report = {"schema_version": 1, "question": request.question,
                "answer_style": request.answer_style, "status": "failed", "stage": "setup",
                "runtime_load_count": 0, "created_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(exc).__name__, "message": str(exc), "stage": "setup"}}
            _persist_setup_report(output, report)
            if progress:
                progress("setup", "failed")
            return report
        # Do not wrap exceptions or rewrite lifecycle flags from the live runner:
        # the service handles unexpected exceptions as unknown cleanup state.
        return pipeline.run_live_question(request.question, output, cache_root,
            tokenizer_loader=lambda: pipeline.load_tokenizer(tokenizer_dir),
            encoder_info=policy,
            encoder_factory=lambda: pipeline.BGEEncoder(model_dir, tokenizer_dir, device="mps"),
            answer_tokenizer_loader=lambda: pipeline.load_answer_tokenizer(root, "4b"),
            runtime_factory=make_runtime, client=client, limit=request.limit, pubmed_query=request.pubmed_query,
            generate=request.generate, top_k=request.top_k, answer_style=request.answer_style,
            progress=progress, release_accelerator=_release_mps)

    return run
