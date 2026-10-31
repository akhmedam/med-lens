"""Run a question against one verified local collection, preserving every stage."""

import gc
import hashlib
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from medlens.generation.answers import generate_answer
from medlens.generation.context import build_context, digest
from medlens.inference.qwen import MODEL_ID, REVISION
from medlens.reranking.listwise import apply_response, build_prompt


def tokenizer_identity(tokenizer):
    return {"model_id": MODEL_ID, "revision": REVISION,
        "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "special_tokens": False, "payload_format": "canonical-json-sources-v1"}


def run_question(index, question, *, encoder_factory, tokenizer, runtime_factory,
                 generate=False, rerank=False, top_k=10, checkpoint=None,
                 release_accelerator=None):
    """Factories keep BGE and Qwen lifetimes sequential; Qwen is loaded at most once."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty research question is required")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be between one and ten")
    if type(generate) is not bool or type(rerank) is not bool:
        raise ValueError("generate and rerank must be booleans")
    info = tokenizer_identity(tokenizer)
    implementation = {name: hashlib.sha256(Path(__file__).parent.joinpath(name).read_bytes()).hexdigest()
        for name in ("pipeline.py", "reporting.py", "retrieval/dense.py", "retrieval/embeddings.py",
            "reranking/listwise.py", "inference/qwen.py", "generation/context.py",
            "generation/answers.py", "generation/quotes.py")}
    identity = {"question": question, "dataset_id": index.dataset.manifest["dataset_id"],
        "window_artifact_id": index.window_manifest["artifact_id"],
        "embedding_artifact_id": index.embedding_manifest["artifact_id"],
        "top_k": top_k, "generate": generate, "rerank": rerank, "tokenizer": info,
        "implementation_sha256": implementation}
    report = {"schema_version": 1, "request_id": digest(identity), **identity,
        "status": "running", "stage": "retrieval", "timings": {},
        "scope": {"source": "provided-local-collection", "candidate_count": len(index.articles),
            "searchable_count": len(index.articles) - len(index.missing_pmids),
            "window_count": len(index.windows), "missing_vector_pmids": index.missing_pmids},
        "runtime_load_count": 0}
    started = perf_counter()
    runtime = None

    def save(stage):
        report["stage"] = stage
        if checkpoint:
            checkpoint(report)

    def model():
        nonlocal runtime
        if runtime is None:
            runtime = runtime_factory()
            report["runtime_load_count"] += 1
            if tokenizer_identity(runtime.tokenizer) != info:
                raise ValueError("Runtime and evidence tokenizers differ")
            report["runtime"] = runtime.metadata
        return runtime

    try:
        save("retrieval")
        tick = perf_counter()
        encoder = encoder_factory()
        try:
            index.verify_encoder(encoder)
            vectors, encoding = encoder.encode_queries([question])
            if vectors.shape != (1, 768):
                raise ValueError("Expected exactly one BGE query vector")
            hits = index.search(vectors[0], top_k)
            scores = np.clip(index.vectors @ vectors[0], -1.0, 1.0)
            report["retrieval"] = {"method": "dense-cosine-max-window", "results": hits,
                "query_vector_sha256": hashlib.sha256(vectors[0].astype(np.float32).tobytes()).hexdigest(),
                "encoder": encoder.info, "encoding": encoding}
        finally:
            del encoder
            gc.collect()
            if release_accelerator:
                release_accelerator()
        report["timings"]["retrieval_seconds"] = perf_counter() - tick
        save("retrieval-complete")

        ranking = [h["pmid"] for h in hits]
        tick = perf_counter()
        save("context")
        # Pack dense evidence before any optional model load, so a Qwen failure
        # still leaves inspectable source passages in the checkpoint.
        report["context"] = build_context(index.dataset, index.window_manifest, index.windows,
            ranking, scores, tokenizer, info)
        report["timings"]["context_seconds"] = perf_counter() - tick
        save("context-complete")

        tick = perf_counter()
        save("reranking")
        if rerank and len(ranking) > 1:
            instance = model()
            prompt = build_prompt(question, [index.articles[p] for p in ranking], instance.tokenizer)
            generated = instance.generate(prompt, max_new_tokens=128)
            result = apply_response(generated["text"], prompt["mapping"],
                hit_output_limit=generated["hit_output_limit"])
            report["reranking"] = {"status": "fallback" if result["fallback"] else "applied",
                "before_ranking": ranking, **result, "prompt": prompt, "generation": generated}
            ranking = result["ranking"]
        else:
            report["reranking"] = {"status": "skipped", "reason": "not-requested" if not rerank else "fewer-than-two-hits",
                "ranking": ranking}
        report["timings"]["reranking_seconds"] = perf_counter() - tick
        save("reranking-complete")

        if ranking != report["context"]["article_ranking"]:
            tick = perf_counter()
            save("context")
            report["context"] = build_context(index.dataset, index.window_manifest, index.windows,
                ranking, scores, tokenizer, info)
            report["timings"]["context_seconds"] += perf_counter() - tick
            save("context-complete")

        tick = perf_counter()
        save("answer")
        if generate:
            # An empty context can abstain without constructing a language model.
            instance = model() if report["context"]["sources"] else SimpleNamespace(tokenizer=tokenizer)
            report["answer"] = generate_answer(question, report["context"], instance)
        else:
            report["answer"] = {"status": "not_requested", "claims": [], "reason": "generation-disabled"}
        report["timings"]["answer_seconds"] = perf_counter() - tick
        report["status"] = "complete"
        save("complete")
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "stage": report["stage"]}
        # Preserve completed retrieval/context even when loading or running Qwen fails.
    finally:
        report["timings"]["pipeline_seconds"] = perf_counter() - started
        if checkpoint:
            checkpoint(report)
    return report
