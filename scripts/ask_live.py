"""Fetch fresh PubMed candidates and generate a sourced answer in one local request."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

from medlens.inference.profiles import load_answer_tokenizer
from medlens.inference.backends import add_runtime_arguments, runtime_factory
from medlens.live import local_encoder_policy, run_live_question
from medlens.pipeline import ANSWER_STYLES
from medlens.pubmed_query import plan_query
from medlens.retrieval.bge_tokenizer import load_tokenizer
from medlens.retrieval.embeddings import BGEEncoder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query', required=True, help='English biomedical research question')
    parser.add_argument('--pubmed-query', help='Exact PubMed expression to use instead of the question; no topic filter is appended')
    parser.add_argument('--limit', type=int, default=100, help='Fresh PubMed candidates, 1–300; default 100')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--answer-style', choices=ANSWER_STYLES, default='draft',
        help='Answer format: draft (default) or experimental model-selected source quotations')
    parser.add_argument('--search-only', action='store_true', help='Diagnostic search/evidence without Qwen inference')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache/embeddings/bge-input-v1')
    parser.add_argument('--device', choices=('mps','cpu'), default='mps', help='BGE device only; Qwen uses the Apple GPU')
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    try:
        make_runtime = runtime_factory(args.engine, model=args.answer_model, root=ROOT, model_dir=args.mlx_model_dir,
                                       worker_python=args.mlx_python)
    except ValueError as exc:parser.error(str(exc))
    try:
        plan_query(args.query, args.pubmed_query)
        if not 1 <= args.limit <= 300 or not 1 <= args.top_k <= 10:raise ValueError('Invalid candidate limit or top-k')
        if args.output_dir.exists():raise ValueError('Choose a fresh output directory')
        policy = local_encoder_policy(ROOT/'.cache/models/bge-base-en-v1.5', device=args.device)
    except (ValueError, OSError) as exc:parser.error(str(exc))
    def release():
        import torch
        if torch.backends.mps.is_available():torch.mps.empty_cache()
    def answer_tokenizer():
        return load_answer_tokenizer(ROOT, args.answer_model)
    last = None
    def progress(stage, status):
        nonlocal last
        if (stage,status) != last:print(stage,status,flush=True);last=(stage,status)
    result = run_live_question(args.query, args.output_dir, args.cache,
        tokenizer_loader=lambda:load_tokenizer(ROOT/'.cache/tokenizers/bge-base-en-v1.5'),
        encoder_info=policy, encoder_factory=lambda:BGEEncoder(ROOT/'.cache/models/bge-base-en-v1.5',
            ROOT/'.cache/tokenizers/bge-base-en-v1.5', device=args.device),
        answer_tokenizer_loader=answer_tokenizer, runtime_factory=make_runtime,
        limit=args.limit, pubmed_query=args.pubmed_query, generate=not args.search_only,
        top_k=args.top_k, answer_style=args.answer_style, progress=progress, release_accelerator=release)
    print((args.output_dir/'report.html').resolve())
    if result['status']!='complete':return 130 if result['status']=='interrupted' else 1
    return 2 if result['answer']['status']=='validation_failed' else 0


if __name__=='__main__':raise SystemExit(main())
