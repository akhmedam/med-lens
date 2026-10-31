"""Search a verified local collection and optionally generate a cited research draft."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from medlens.inference.qwen import MODEL_ID, REVISION, QwenRuntime
from medlens.pipeline import run_question
from medlens.reporting import render_report
from medlens.retrieval.dense import DenseIndex
from medlens.retrieval.embeddings import BGEEncoder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--query', required=True, help='Research question in English')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--generate', action='store_true', help='Enable experimental Qwen draft generation')
    parser.add_argument('--rerank', action='store_true', help='Rerank dense hits with the same Qwen instance')
    parser.add_argument('--model', type=Path, default=ROOT / '.cache/models/bge-base-en-v1.5')
    parser.add_argument('--tokenizer', type=Path, default=ROOT / '.cache/tokenizers/bge-base-en-v1.5')
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps', help='BGE device; Qwen requires MPS')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.query.strip() or not 1 <= args.top_k <= 10:
        parser.error('Supply a nonempty query and top-k between one and ten')
    # A dedicated directory owns its checkpoints; never overwrite an existing run.
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('Output directory already exists; choose a fresh path')
    started = perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    report = {'schema_version': 1, 'question': args.query, 'status': 'running', 'stage': 'setup'}
    paths = {name: str(getattr(args, name).resolve()) for name in ('dataset', 'windows', 'embeddings')}
    last_state = None
    def checkpoint(value):
        nonlocal last_state
        value.update(created_at=created, input_paths=paths, request_seconds=perf_counter() - started)
        for filename, body in (('run.json', json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                               ('report.html', render_report(value))):
            temporary = args.output_dir / (filename + '.tmp')
            temporary.write_text(body, encoding='utf-8')
            temporary.replace(args.output_dir / filename)
        state = (value['stage'], value['status'])
        if state != last_state:
            print(*state, flush=True)
            last_state = state
    checkpoint(report)
    try:
        index = DenseIndex(args.dataset, args.windows, args.embeddings)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, token=False,
            trust_remote_code=False, local_files_only=True)
        def release_accelerator():
            if args.device == 'mps':
                import torch
                torch.mps.empty_cache()
        report = run_question(index, args.query,
            encoder_factory=lambda: BGEEncoder(args.model, args.tokenizer, device=args.device),
            tokenizer=tokenizer, runtime_factory=QwenRuntime, generate=args.generate,
            rerank=args.rerank, top_k=args.top_k, checkpoint=checkpoint,
            release_accelerator=release_accelerator)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            error={'type': type(exc).__name__, 'message': str(exc), 'stage': report['stage']})
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    checkpoint(report)
    print(str((args.output_dir / 'report.html').resolve()))
    if report['status'] != 'complete':
        return 130 if report['status'] == 'interrupted' else 1
    return 2 if report.get('answer', {}).get('status') == 'validation_failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
