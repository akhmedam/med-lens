"""Prepare source-bound evidence contexts for frozen pilot article rankings."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from rescore_pilot import rescore
from medlens.evaluation.formats import parse_queries
from medlens.generation.context import build_context, locate_quote, digest
from medlens.retrieval.dense import DenseIndex
from medlens.retrieval.embeddings import BGEEncoder
from medlens.reranking.qwen import MODEL_ID, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "benchmarks/pilot-v1/evaluation.json")
    parser.add_argument("--method", choices=("dense", "rrf", "dense_qwen", "rrf_qwen"), default="dense_qwen")
    parser.add_argument("--windows-root", type=Path, default=ROOT / "data/windows/pilot-v1-002")
    parser.add_argument("--embeddings-root", type=Path, default=ROOT / "data/embeddings/pilot-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.summary.resolve() or args.output.exists() or args.summary.exists():
        parser.error("Choose two distinct unused output paths")
    evaluation, _, _ = rescore(args.profile)
    profile = json.loads(args.profile.read_bytes())
    query_path = ROOT / profile["queries"]["path"]
    queries = parse_queries(query_path.read_bytes(), query_path.parent)
    by_id = {q.query_id: q for q in queries}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True,
        trust_remote_code=False, token=False)
    tokenizer_info = {"model_id": MODEL_ID, "revision": REVISION,
        "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "special_tokens": False, "payload_format": "canonical-json-sources-v1"}
    encoder = BGEEncoder(ROOT / ".cache/models/bge-base-en-v1.5", ROOT / ".cache/tokenizers/bge-base-en-v1.5")
    vectors, encoding = encoder.encode_queries([by_id[r["query_id"]].text for r in evaluation["per_query"]])
    rows, public = [], []
    started = time.perf_counter()
    for row, vector in zip(evaluation["per_query"], vectors):
        query = by_id[row["query_id"]]
        index = DenseIndex(query.dataset, args.windows_root / query.query_id, args.embeddings_root / query.query_id)
        if index.dataset.manifest["dataset_id"] != row["dataset_id"]:
            raise ValueError("Frozen ranking and current dense dataset differ")
        index.verify_encoder(encoder)
        # Reuse dense query validation before calculating the full window score array.
        index.search(vector, top_k=1)
        import numpy as np
        scores = np.clip(index.vectors @ vector, -1.0, 1.0)
        context = build_context(index.dataset, index.window_manifest, index.windows,
            row["methods"][args.method]["ranking"], scores, tokenizer, tokenizer_info)
        for source in context["sources"]:
            location = locate_quote(context, source["citation_id"], source["text"])
            assert location["matches"] == [{"char_start": source["char_start"], "char_end": source["char_end"]}]
        rows.append({"query_id": query.query_id, "query": query.text,
            "query_vector_sha256": hashlib.sha256(vector.astype(np.float32).tobytes()).hexdigest(),
            "context": context})
        public.append({"query_id": query.query_id, "context_id": context["context_id"],
            "source_count": len(context["sources"]), "article_count": len({s["pmid"] for s in context["sources"]}),
            "evidence_tokens": context["evidence_tokens"], "status": context["status"],
            "selected_window_ids": [s["window_id"] for s in context["sources"]],
            "skips": {reason: sum(s["reason"] == reason for s in context["skipped"])
                      for reason in sorted({s["reason"] for s in context["skipped"]})}})
        print(query.query_id, len(context["sources"]), context["evidence_tokens"], flush=True)
    identity = {"schema_version": 1, "evaluation_run_id": evaluation["run_id"],
        "profile_sha256": evaluation["profile_sha256"], "qrels_sha256": evaluation["qrels_sha256"],
        "method": args.method, "context_ids": [r["context"]["context_id"] for r in rows],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    summary = {**identity, "run_id": digest(identity), "query_count": len(rows),
        "selection_and_validation_seconds": time.perf_counter() - started,
        "context_config": rows[0]["context"]["config"],
        "selection_policy": rows[0]["context"]["selection_policy"],
        "context_implementation_sha256": rows[0]["context"]["implementation_sha256"],
        "tokenizer": tokenizer_info, "query_encoder": encoder.info, "query_encoding": encoding,
        "per_query": public, "assertions": {"source_spans_match": True, "full_excerpt_quote_checks": True,
            "generation_run": False, "claim_entailment_checked": False}}
    report = {**summary, "per_query": rows}
    for path, value in ((args.output, report), (args.summary, summary)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
